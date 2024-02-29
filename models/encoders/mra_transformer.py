
import math
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)


import torch
import torch.nn as nn
from mra_helper import OverlapPatchEmbed, Block, PosCNN, PatchEmbed, PatchMerging
# import sys
# sys.path.append('..')
from configs.config_imagenet import config as cfg

from timm.models.layers import trunc_normal_
from functools import partial
from utils.logger import get_logger




# How to apply multihead multiscale
class MRATransformer(nn.Module):
    def __init__(self, img_size=(1024, 1024), patch_size=4, in_chans=3, num_classes=1000, embed_dims=[96, 192, 384, 768], 
                 num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, local_region_scales=[3, 3, 2, 1], 
                 depths=[2, 2, 6, 2]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.logger = get_logger()
        self.img_size = img_size
        # print('img_size: ',img_size)

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size = self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_layer=norm_layer)
        self.patch_embed2 = PatchMerging(img_size=(img_size[0]// 4,img_size[1]//4), in_chans=embed_dims[0], norm_layer=norm_layer)
        self.patch_embed3 = PatchMerging(img_size=(img_size[0]//8, img_size[1]//8), in_chans=embed_dims[1], norm_layer=norm_layer)
        self.patch_embed4 = PatchMerging(img_size=(img_size[0]//16, img_size[1]//16), in_chans=embed_dims[2], norm_layer=norm_layer)
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # print(f'dpr: {dpr}')
        # 56x56
        
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            n_local_region_scales=local_region_scales[0], img_size=(img_size[0]// 4,img_size[1]//4))
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        # 28x28
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            n_local_region_scales=local_region_scales[1], img_size=(img_size[0]//8, img_size[1]//8))
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        # 14x14
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            n_local_region_scales=local_region_scales[2], img_size=(img_size[0]// 16,img_size[1]//16))
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        #7x7
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            n_local_region_scales=local_region_scales[3], img_size=(img_size[0]// 32,img_size[1]//32))
            for i in range(depths[3])])             
        self.norm4 = norm_layer(embed_dims[3])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[3], num_classes) if self.num_classes > 0 else nn.Identity()

        # cur += depths[3]

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            self.load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def load_dualpath_model(self, model, model_file):
    # load raw state_dict
        t_start = time.time()
        if isinstance(model_file, str):
            raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            #raw_state_dict = torch.load(model_file)
            if 'model' in raw_state_dict.keys():
                raw_state_dict = raw_state_dict['model']
        else:
            raw_state_dict = model_file
        

        t_ioend = time.time()

        model.load_state_dict(raw_state_dict, strict=False)
        #del state_dict
        t_end = time.time()
        self.logger.info("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(t_ioend - t_start, t_end - t_ioend))

    def forward_features(self, x_rgb):
        """
        x_rgb: B x N x H x W
        """
        B = x_rgb.shape[0]
        # stage 1
        stage = 0
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # print('Stage 1 - Tokenization: {}'.format(x_rgb.shape))
        for j,blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        # print('########### Stage 1 - Output: {}'.format(x_rgb.shape))
        x_rgb = self.norm1(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        self.logger.info('Stage 1 - Output: {}'.format(x_rgb.shape))
        # print('########### Stage 1 - Output: {}'.format(x_rgb.shape))

        # stage 2
        stage += 1
        x_rgb, H, W = self.patch_embed2(x_rgb)
        self.logger.info('Stage 2 - Tokenization: {}'.format(x_rgb.shape))
        for j,blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm2(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        self.logger.info('Stage 2 - Output: {}'.format(x_rgb.shape))
        # print('############# Stage 2 - Output: {}'.format(x_rgb.shape))

        # stage 3
        stage += 1
        x_rgb, H, W = self.patch_embed3(x_rgb)
        self.logger.info('Stage 3 - Tokenization: {}'.format(x_rgb.shape))
        for j,blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm3(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        self.logger.info('Stage 3 - Output: {}'.format(x_rgb.shape))
        # print('###########Stage 3 - Output: {}'.format(x_rgb.shape))

        # stage 4
        stage += 1
        x_rgb, H, W = self.patch_embed4(x_rgb)
        self.logger.info('Stage 4 - Tokenization: {}'.format(x_rgb.shape))
        # print('Stage 4 - Tokenization: {}'.format(x_rgb.shape))
        for j,blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm4(x_rgb)   # B, L, C
        # x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        self.logger.info('Stage 4 - Output: {}'.format(x_rgb.shape))
        # print('########## Stage 4 - Output: {}'.format(x_rgb.shape))

        x = x_rgb.transpose(1,2).contiguous()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        cls_output = self.head(x)

        # B, 768, 7, 7
        return x_rgb, cls_output

    def forward(self, x_rgb):
        # print()
        out = self.forward_features(x_rgb)
        return out

    def flops(self):
        flops = 0
        flops += self.patch_embed1.flops()
        flops += self.patch_embed2.flops()
        flops += self.patch_embed3.flops()
        flops += self.patch_embed4.flops()

        for i, blk in enumerate(self.block1):
            flops += blk.flops()
        for i, blk in enumerate(self.block2):
            flops += blk.flops()
        for i, blk in enumerate(self.block3):
            flops += blk.flops()
        for i, blk in enumerate(self.block4):
            flops += blk.flops()
        
        return flops


class mit_b0(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        img_size = (fuse_cfg.IMAGE.image_height, fuse_cfg.IMAGE.image_width)
        heads = fuse_cfg.MODEL.heads    #3,6,12,24
        super(mit_b0, self).__init__(
            img_size = img_size, patch_size = 4, num_classes=fuse_cfg.DATASET.NUM_CLASSES, embed_dims=[96, 192, 384, 768], 
            num_heads=heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), local_region_scales = [3, 3, 2, 1], depths=[2, 2, 6, 2], 
            drop_rate=fuse_cfg.MODEL.DROP_RATE, drop_path_rate=fuse_cfg.MODEL.DROP_PATH_RATE)


class mit_b1(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        img_size = (fuse_cfg.IMAGE.image_height, fuse_cfg.IMAGE.image_width)
        heads = fuse_cfg.MODEL.heads
        super(mit_b1, self).__init__(
            img_size = img_size, patch_size=4, embed_dims=[64, 128, 320, 512], 
            num_heads=heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
            drop_rate=fuse_cfg.MODEL.DROP_RATE, drop_path_rate=0.1)


class mit_b2(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        img_size = (fuse_cfg.IMAGE.image_height, fuse_cfg.IMAGE.image_width)
        heads = fuse_cfg.MODEL.heads
        super(mit_b2, self).__init__(
            img_size=img_size, patch_size=4, embed_dims=[64, 128, 320, 512], 
            num_heads=heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], 
            drop_rate=fuse_cfg.MODEL.DROP_RATE, drop_path_rate=0.1)



if __name__=="__main__":
    backbone = mit_b0(fuse_cfg=cfg, norm_layer = nn.BatchNorm2d)
    
    # ########print(backbone)
    B = 8
    C = 3
    H = 224
    W = 224
    device = 'cuda:1'
    rgb = torch.randn(B, C, H, W)
    outputs = backbone(rgb)
    print(f'outputs:{outputs[0].size()} cls:{outputs[1].size()}')

    # Assuming 'model' is your PyTorch model
    total_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")