import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time


class WindowSEBlock(nn.Module):
    def __init__(self, num_windows, reduction=16):
        super(WindowSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Adapt for pooling across each window
        self.fc = nn.Sequential(
            nn.Linear(num_windows, num_windows // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_windows // reduction, num_windows, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: BxCxNxS, where N is number of windows, S is spatial dimension (flattened)
        b, c, n, s = x.shape
        # Reshape to treat each window equally and merge C and S for pooling
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(b, n, -1)  # Shape: BxNx(C*S)
        # Pooling across the combined C*S dimension to get BxNx1
        y = self.avg_pool(x_reshaped.view(b, n, c, s)).view(b, n)  # Now: BxN
        # Pass through FC to get recalibration weights
        y = self.fc(y).view(b, n, 1, 1)
        # Apply recalibration across windows
        recalibrated = x * y.expand(-1, -1, c, s).permute(0, 2, 1, 3)  # Expand and apply to original shape
        
        return recalibrated


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., window_size=7, reduction=4, img_size=(32, 32)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        
        self.H, self.W = img_size[0], img_size[1]
        self.N_G = self.H//self.window_size * self.W//self.window_size

        
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Linear embedding
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias) 
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reduction = reduction

        if self.reduction > 0:
            self.SE_region = WindowSEBlock(num_windows=self.N_G, reduction=reduction)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

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

    
    """ arr.shape: B, N, C"""
    # create non-overlapping windows of size=self.window_size
    # input arr --> (Bx lC X H x W 
    # output patches--> (BxNw) x (self.window_size^2) x lC/lH

    def window_partition(self, arr):
        # print(arr.shape)
        B = arr.shape[0]
        H = arr.shape[1]
        W = arr.shape[2]
        C = arr.shape[3]
        
        arr_reshape = arr.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        arr_perm = arr_reshape.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)       
        return arr_perm


    def attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        x = (attn @ v)
        # print(f'attn:{attn.shape} v:{v.shape}')
        return x, attn
    



    def forward(self, x, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        # N = H*W
        self.H=H
        self.W=W
        A = []
        B, N, C = x.shape
        # print('reshape: ',x.shape)
        assert N==self.H*self.W
        x = x.view(B, H, W, C)
        x_windows = self.window_partition(x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        B_, Nr, C = x_windows.shape     # B_ = B * num_local_regions
        # print('x_windows: ',x_windows.shape)

        qkv = self.qkv_proj(x).reshape(B_, Nr, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(f'q:{q.shape} k:{k.shape} v:{v.shape}')

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, Nr, C).view(B, self.N_G, Nr, C)

        if self.reduction > 0:
            x_perm = x.permute(0, 3, 1, 2).contiguous()
            x = self.SE_region(x_perm)
            # print(f'output SE:{x.shape} ')
            x = x.permute(0, 2, 3, 1).contiguous()
            # print(f'output SE perm:{x.shape} ')
            
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def flops(self):
        # FLOPs for linear layers
        flops_linear_q = 2 * self.dim * self.dim
        flops_linear_kv = 2 * self.dim * self.dim * 2
        head_dim = self.dim // self.num_heads
        flops = 0
        print("number of heads ", self.num_heads)
        for i in range(self.num_heads):
            r_size = self.local_region_shape[i]
            if r_size == 1:
                N = self.H * self.W
                flops_attention_weight = N * head_dim * N
                flops_attention_output = N * N * head_dim

            else:
                region_number = (self.H * self.W) // (r_size ** 2)
                p = r_size ** 2
                flops_attention_weight = region_number * (p * head_dim * p)
                flops_attention_output = region_number * (p * p * head_dim)
            flops_head = flops_attention_weight + flops_attention_output
            flops += flops_head    

        total_flops = flops_linear_q + flops_linear_kv + flops
        return total_flops


        
if __name__=="__main__":
    # #######print(backbone)
    B = 4
    C = 96
    H = 56
    W = 56
    # device = 'cuda:1'
    ms_attention = MultiScaleAttention(C, num_heads=3, window_size=7, img_size=(H, W))
    # ms_attention = ms_attention.to(device)
    # # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('final output: ',y.shape)