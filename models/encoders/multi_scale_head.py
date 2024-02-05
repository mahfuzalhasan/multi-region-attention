import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time



class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., n_local_region_scales = 3, window_size=7, img_size=(32, 32)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.n_local_region_scales = n_local_region_scales
        self.img_size = img_size
        self.local_dim = self.dim//self.n_local_region_scales
        self.local_head = self.num_heads//self.n_local_region_scales
        self.H, self.W = img_size[0], img_size[1]
        self.N_G = self.H//self.window_size * self.W//self.window_size

        assert self.num_heads%n_local_region_scales == 0
        # Linear embedding
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias) 
        # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

        downsample_layers = []
        upsample_layers = []
        if self.n_local_region_scales > 1:
            for group_no in range(1, self.n_local_region_scales):
                stride = int(math.pow(2, group_no-1))
                padding = group_no
                dilation = group_no
                conv = nn.Sequential(
                                nn.Conv2d(self.local_dim, self.local_dim, kernel_size=3, stride=stride, padding=padding, dilation=dilation, groups=head_dim),
                                nn.MaxPool2d(2, 2)
                            )
                upconv = nn.ConvTranspose2d(self.local_dim, self.local_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
                downsample_layers.append(conv)
                upsample_layers.append(upconv)

        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        
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
    def windowing(self, arr):
        # print(arr.shape)
        B = arr.shape[0]
        local_C = arr.shape[1]
        H = arr.shape[2]
        W = arr.shape[3]
        
        arr_reshape = arr.view(B, local_C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        arr_perm = arr_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, self.window_size* self.window_size, local_C)
        arr_perm = arr_perm.reshape(-1, self.window_size* self.window_size, self.local_head, local_C // self.local_head).permute(0, 2, 1, 3).contiguous()        
        return arr_perm


    def attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        x = (attn @ v)
        # print(f'attn:{attn.shape} v:{v.shape}')
        return x, attn
    def upsample(self, x, i):
        B, h, r, N, l_C = x.shape
        perm_x = x.reshape(B, h*l_C, r, 7, 7).permute(0, 1, 3, 4, 2).contiguous().reshape(B, h*l_C, 7, 7*r)
        up_x = self.upsample_layers[i-1](perm_x)
        upsampled_x = F.interpolate(up_x, size=(7, self.N_G*7), mode='nearest')
        upsampled_x = upsampled_x.view(B, h, l_C, 7, 7, self.N_G).permute(0, 1, 5, 3, 4, 2).contiguous().reshape(B, h, self.N_G, -1, l_C)
        return upsampled_x



    def forward(self, x, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        # N = H*W
        self.H=H
        self.W=W
        A = []
        B, N, C = x.shape
        assert N==self.H*self.W

        temp = self.qkv_proj(x).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2)
        
        self.attn_outcome_per_group = []
        self.attn_mat_per_head = []
        
        for i in range(self.n_local_region_scales):
            local_C = C//self.n_local_region_scales
            qkv = temp[:, :, i*local_C:i*local_C + local_C, :, :]
            if i > 0:
                qkv = qkv.reshape(B*3, local_C, H, W)
                qkv = self.downsample_layers[i-1](qkv)
                qkv = qkv.reshape(B, 3, local_C, self.H//pow(2,i), self.W//pow(2,i))
            qkv = qkv.permute(1, 0, 2, 3, 4)
            q,k,v = qkv[0], qkv[1], qkv[2]      #B, l_C, H, W
        
            q = self.windowing(q)
            k = self.windowing(k)
            v = self.windowing(v)
            
            output_size = (self.H//pow(2,i), self.W//pow(2,i))
            n_region = (output_size[0]//self.window_size) * (output_size[1]//self.window_size)
            
            y, attn = self.attention(q, k, v)
            y = y.reshape(B, n_region, self.local_head, self.window_size* self.window_size, local_C // self.local_head).permute(0, 2, 1, 3, 4)
            if i>0:
                y = self.upsample(y, i)
            self.attn_outcome_per_group.append(y)

        # # #concatenating multi-group attention
        attn_fused = torch.cat(self.attn_outcome_per_group, axis=1)
        # print('concatL ',attn_fused.shape)
        attn_fused = attn_fused.reshape(B, self.num_heads, -1, C//self.num_heads)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused )
        return attn_fused

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
    C = 768
    H = 7
    W = 7
    # device = 'cuda:1'
    ms_attention = MultiScaleAttention(C, num_heads=24, n_local_region_scales=1, window_size=7)
    # ms_attention = ms_attention.to(device)
    # # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('output: ',y.shape)