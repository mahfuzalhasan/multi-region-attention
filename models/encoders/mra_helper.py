import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

from se_head_attn import MultiScaleAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from configs.config_imagenet import config


import math
import time
from utils.logger import get_logger

logger = get_logger()




class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.dim = dim

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B N C -> B C N -> B C H W
        self.input_shape = x.shape
        x = self.dwconv(x) 
        x = x.flatten(2).transpose(1, 2) # B C H W -> B N C

        return x

    def flops(self):
        # Correct calculation for output dimensions
        padding = (1,1) 
        kernel_size = (3,3)
        stride = 1
        groups = self.dim
        in_chans = self.dim
        out_chans = self.dim

        output_height = ((self.input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride) + 1
        output_width = ((self.input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride) + 1

        # Convolution layer FLOPs
        conv_flops = 2 * out_chans * output_height * output_width * kernel_size[0] * kernel_size[1] * in_chans / groups

        total_flops = conv_flops
        return total_flops



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        # x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def flops(self):
        # H, W = self.H, self.W
        flops_mlp = self.fc1.in_features * self.fc1.out_features * 2
        flops_mlp += self.dwconv.flops()
        flops_mlp += self.fc2.in_features * self.fc2.out_features * 2
        return flops_mlp

class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, reduction=4, img_size=(1024, 1024)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.reduction = reduction
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn = MultiScaleAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, reduction=self.reduction, img_size=img_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
    
    def flops(self):
        # FLOPs for MultiScaleAttention
        attn_flops = self.attn.flops()

        # FLOPs for Mlp
        mlp_flops = self.mlp.flops()

        total_flops = attn_flops + mlp_flops
        return total_flops


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.input_shape = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
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

    def forward(self, x):
        # B C H W
        self.input_shape = x.shape
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)
        return x, H, W

    def flops(self):
        # Correct calculation for output dimensions
        padding = (self.patch_size[0] // 2, self.patch_size[1] // 2)
        output_height = ((self.input_shape[2] + 2 * padding[0] - self.patch_size[0]) // self.stride) + 1
        output_width = ((self.input_shape[3] + 2 * padding[1] - self.patch_size[1]) // self.stride) + 1

        # Convolution layer FLOPs
        conv_flops = 2 * self.embed_dim * output_height * output_width * self.patch_size[0] * self.patch_size[1] * self.in_chans

        # Layer normalization FLOPs
        norm_flops = 2 * self.embed_dim * output_height * output_width

        total_flops = conv_flops + norm_flops
        return total_flops
    
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        use_conv_embed (bool): Wherther use overlapped convolutional embedding layer. Default: False.
        norm_layer (nn.Module, optional): Normalization layer. Default: None 
        use_pre_norm (bool): Whether use pre-normalization before projection. Default: False
        is_stem (bool): Whether current patch embedding is stem. Default: False
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, 
                    use_conv_embed=False, norm_layer=None, use_pre_norm=False, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm
        self.use_conv_embed = use_conv_embed

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.kernel_size = kernel_size
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        

        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        if self.use_conv_embed:
            flops = Ho * Wo * self.embed_dim * self.in_chans * (self.kernel_size**2)
        else:
            flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]