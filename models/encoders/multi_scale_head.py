import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time

class MergeRegions(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads=3):
        super(MergeRegions, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel * num_heads, 
                              out_channels=out_channel * num_heads, 
                              kernel_size=2, stride=2, groups=num_heads)
    
    # B, num_local_head, num_regions_7x7, 49(7x7 flattened), head_dim 
    def forward(self, x):
        B, h, R, Nr, C_h = x.shape
        R_out = R//4
        r = int(math.sqrt(R))
       
        x = x.view(B, h * C_h, R, Nr)
        x_reshaped = x.view(B, h * C_h, r, r * Nr)
        x_merged = self.conv(x_reshaped)
        
        out_shape = (B, h, R_out, Nr, C_h)
        x_out = x_merged.view(out_shape)
        return x_out

class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., n_local_region_scales = 3, window_size=7, img_size=(32, 32)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        self.window_size = window_size
        
        self.n_local_region_scales = n_local_region_scales
        self.local_dim = self.dim//self.n_local_region_scales
        self.local_head = self.num_heads//self.n_local_region_scales
        
        self.img_size = img_size
        self.H, self.W = img_size[0], img_size[1]
        self.N_G = (self.H//self.window_size) * (self.W//self.window_size)

        assert self.num_heads%n_local_region_scales == 0
        # Linear embedding
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias) 
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), self.local_head))  # 2*Wh-1 * 2*Ww-1, nH
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
        trunc_normal_(self.relative_position_bias_table, std=.02)

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

    
    """ arr.shape: B, H, W, C"""
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
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # scaling needs to be fixed
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x, attn
    
    def downsample(self, x, kernel_size):
        x1 = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
        x2 = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
        return x1+x2

    def forward(self, x, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        # N = H*W
        self.H=H
        self.W=W
        A = []
        B, N, C = x.shape

        assert N==self.H*self.W
        x = x.view(B, H, W, C)

        # temp--> 3, B, N, C
        temp = self.qkv_proj(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)

        self.attn_outcome_per_group = []
        self.attn_mat_per_head = []
        
        for i in range(self.n_local_region_scales):
            # print(f'###########i:{i}####################')
            local_C = C//self.n_local_region_scales
            # 3, B, N, local_C
            qkv = temp[:, :, :, i*local_C:i*local_C + local_C]

            output_size = (self.H//pow(2,i), self.W//pow(2,i))
            n_region = (output_size[0]//self.window_size) * (output_size[1]//self.window_size)
            qkv = qkv.reshape(-1, N, local_C).permute(0,2,1).contiguous().reshape(3*B, local_C, self.H, self.W)
            if i>0:
                ########## simple downsampling --> 2D kernel
                qkv = self.downsample(qkv, kernel_size=int(math.pow(2,i)))
                ##############################
            qkv = qkv.permute(0, 2, 3, 1).contiguous()
            qkv = self.window_partition(qkv)
            qkv = qkv.reshape(3, B, n_region, self.window_size, self.window_size, self.local_head, self.head_dim).permute(0, 1, 5, 2, 3, 4, 6).contiguous()
            q,k,v = qkv[0], qkv[1], qkv[2]  #B, n_region, h, ws, ws, Ch
            q,k,v = q.reshape(-1, self.local_head, self.window_size*self.window_size, self.head_dim), k.reshape(-1, self.local_head, self.window_size*self.window_size, self.head_dim), v.reshape(-1, self.local_head, self.window_size*self.window_size, self.head_dim)

            y, attn = self.attention(q, k, v)
            
            ### For simple upsample
            y = y.reshape(B, n_region, self.local_head, self.window_size* self.window_size, self.head_dim).permute(0, 2, 4, 1, 3).contiguous()
            y = y.reshape(B, local_C, output_size[0], output_size[1])
            ###################### 
            if i>0:
                y = F.interpolate(y, size=(self.H, self.W), mode='bilinear')
            y = y.view(B, self.local_head, self.head_dim, self.H, self.W).permute(0, 1, 3, 4, 2).contiguous()
            self.attn_outcome_per_group.append(y)

        # # #concatenating multi-group attention
        attn_fused = torch.cat(self.attn_outcome_per_group, axis=1)
        attn_fused = attn_fused.reshape(B, self.num_heads, -1, C//self.num_heads)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused )
        return attn_fused

    def flops(self):
        # # qkv = self.qkv_proj(x): For linear projection
        NW = self.window_size * self.window_size
        N = self.H*self.W

        flops_qkv = N * self.dim * (3 * self.dim)
        
        flops_attention = 0
        flops_downsample = 0
        for i in range(self.n_local_region_scales):
            flops_1_window = 0
            if i>0:
                kernel = int(math.pow(2, i))
                h = self.H// kernel
                w = self.W//kernel
                # avg_pool 2d
                flops_downsample += self.local_dim * h * w * (kernel*kernel + 1)

            N_window = self.N_G//math.pow(4,i)
            # A = q*K.T         q,k shape --> B*num_region_7x7, self.local_head, NW*NW, self.head_dim
            flops_1_window += self.local_head * NW * self.head_dim * NW
            # y = A * v
            flops_1_window += self.local_head * NW * NW * self.head_dim
            
            flops_windows = N_window * flops_1_window
            flops_attention += flops_windows
        
        # projection flops
        flops_proj = N * self.dim * self.dim
        # total
        flops = flops_qkv + flops_attention + flops_downsample + flops_proj
        
        return flops


        
if __name__=="__main__":
    # #######print(backbone)
    B = 4
    C = 128
    H = 56
    W = 56
    # device = 'cuda:1'
    ms_attention = MultiScaleAttention(C, num_heads=4, n_local_region_scales=4, window_size=7, img_size=(56, 56))
    # ms_attention = ms_attention.to(device)
    # # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('output: ',y.shape)