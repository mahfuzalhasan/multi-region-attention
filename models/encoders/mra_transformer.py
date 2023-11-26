
import math
import time
import torch
import torch.nn as nn
from .mra_helper import OverlapPatchEmbed, Block

from timm.models.layers import trunc_normal_
from functools import partial
from utils.logger import get_logger

# How to apply multihead multiscale
class MRATransformer(nn.Module):
    def __init__(self, img_size=(1024, 1024), patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], 
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8,4,2,1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.logger = get_logger()
        # print('img_size: ',img_size)

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # 256x256
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], local_region_shape=[8, 16], img_size=(img_size[0]// 4,img_size[1]//4))
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        # 128x128
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], local_region_shape=[4, 8, 8, 16], img_size=(img_size[0]// 8,img_size[1]//8))
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        # 64x64
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], local_region_shape=[1, 2, 2, 4, 4], img_size=(img_size[0]// 16,img_size[1]//16))
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        #32x32
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], local_region_shape=[1, 1, 1, 1, 2, 2, 2, 2], img_size=(img_size[0]// 32,img_size[1]//32))
            for i in range(depths[3])])             
        self.norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

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
        outs = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        self.logger.info('Stage 1 - Tokenization: {}'.format(x_rgb.shape))
        # print('Stage 1 - Output: {}'.format(x_rgb.shape))

        for blk in self.block1:
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm1(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)
        self.logger.info('Stage 1 - Output: {}'.format(x_rgb.shape))
        # print('Stage 1 - Output: {}'.format(x_rgb.shape))

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        self.logger.info('Stage 2 - Tokenization: {}'.format(x_rgb.shape))

        for blk in self.block2:
            x_rgb = blk(x_rgb, H, W)

        x_rgb = self.norm2(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)
        self.logger.info('Stage 2 - Output: {}'.format(x_rgb.shape))

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        self.logger.info('Stage 3 - Tokenization: {}'.format(x_rgb.shape))

        for blk in self.block3:
            x_rgb = blk(x_rgb, H, W)

        x_rgb = self.norm3(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)
        self.logger.info('Stage 3 - Output: {}'.format(x_rgb.shape))

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        self.logger.info('Stage 4 - Tokenization: {}'.format(x_rgb.shape))

        for blk in self.block4:
            x_rgb = blk(x_rgb, H, W)
        x_rgb = self.norm4(x_rgb)
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_rgb)
        self.logger.info('Stage 4 - Output: {}'.format(x_rgb.shape))
        
        return outs

    def forward(self, x_rgb):
        # print()
        out = self.forward_features(x_rgb)
        return out





class mit_b0(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        img_size = (fuse_cfg.IMAGE.image_height, fuse_cfg.IMAGE.image_width)
        heads = fuse_cfg.MODEL.heads
        super(mit_b0, self).__init__(
            img_size = img_size, patch_size=4, embed_dims=[32, 64, 160, 256], 
            num_heads=heads, mlp_ratios=[4, 4, 4, 4],qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], 
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        img_size = (fuse_cfg.IMAGE.image_height, fuse_cfg.IMAGE.image_width)
        heads = fuse_cfg.MODEL.heads
        super(mit_b1, self).__init__(
            img_size = img_size, patch_size=4, embed_dims=[64, 128, 320, 512], 
            num_heads=heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        img_size = (fuse_cfg.IMAGE.image_height, fuse_cfg.IMAGE.image_width)
        heads = fuse_cfg.MODEL.heads
        super(mit_b2, self).__init__(
            img_size=img_size, patch_size=4, embed_dims=[64, 128, 320, 512], 
            num_heads=heads, mlp_ratios=[4, 4, 4, 4], qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], 
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MRATransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


if __name__=="__main__":
    backbone = mit_b2(norm_layer = nn.BatchNorm2d)
    
    # ########print(backbone)
    B = 4
    C = 3
    H = 512
    W = 512
    device = 'cuda:1'
    rgb = torch.randn(B, C, H, W)
    x = torch.randn(B, C, H, W)
    outputs = backbone(rgb)
    for output in outputs:
        print(output.size())