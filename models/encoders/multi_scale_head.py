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

        assert self.num_heads%n_local_region_scales == 0
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        downsample_layers = []
        if self.n_local_region_scales > 1:
            for group in range(1, self.n_local_region_scales):
                stride = int(pow(2, group))
                padding = group - 1
                if group == 1:
                    dilation = 1
                elif group == 2:
                    dilation = 2
                elif group == 3:
                    dilation = 7
                conv = nn.Conv2d(head_dim, head_dim, kernel_size=2, stride=stride, padding=padding, dilation=dilation, groups=head_dim)
                downsample_layers.append(conv)
        self.downsample_layers = nn.ModuleList(downsample_layers)

        
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
    # input arr --> (Bxh) X H x W x Ch
    # output patches--> (BxhxNp) x (self.window_size^2) x Ch
    def patchify(self, arr):
        # print(arr.shape)
        B_nh = arr.shape[0]
        curr_h = arr.shape[1]
        curr_w = arr.shape[2]
        Ch = arr.shape[3]
        patches = arr.reshape(B_nh, curr_h // self.window_size, self.window_size, curr_w // self.window_size, self.window_size, Ch)
        patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, self.window_size**2, Ch)
        return patches




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
        assert N==self.H*self.W

        group_size = self.num_heads//self.n_local_region_scales
        
        # q,k,v --> B,h,N,Ch
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(f'q:{q.shape} k:{k.shape} v:{v.shape}')
        N_g = self.H//self.window_size * self.W//self.window_size
        self.attn_outcome_per_group = []
        self.attn_mat_per_head = []
        
        for i in range(self.n_local_region_scales):

            q_g = q[:, i*group_size:i*group_size+group_size, :, :]
            k_g = k[:, i*group_size:i*group_size+group_size, :, :]
            v_g = v[:, i*group_size:i*group_size+group_size, :, :]
            # print(f'group:{i}')
            q_g = q_g.reshape(-1, self.H, self.W, C//self.num_heads).permute(0, 3, 1, 2).contiguous()
            k_g = k_g.reshape(-1, self.H, self.W, C//self.num_heads).permute(0, 3, 1, 2).contiguous()
            v_g = v_g.reshape(-1, self.H, self.W, C//self.num_heads).permute(0, 3, 1, 2).contiguous()
            
            ## pooling per group using adaptive avg pool
            output_size = (self.H//pow(2,i), self.W//pow(2,i))
            if i>0:
                q_g = self.downsample_layers[i-1](q_g)
                k_g = self.downsample_layers[i-1](k_g)
                v_g = self.downsample_layers[i-1](v_g)
            ############################################
            n_region = (output_size[0]//self.window_size) * (output_size[1]//self.window_size)

            q_g = q_g.permute(0, 2, 3, 1)
            k_g = k_g.permute(0, 2, 3, 1)
            v_g = v_g.permute(0, 2, 3, 1)

            # print('group --> q,k,v normal: ',i, ":", q_g.shape, k_g.shape, v_g.shape)
            q_g = self.patchify(q_g)
            k_g = self.patchify(k_g)
            v_g = self.patchify(v_g)

            # print('group: ',i)
            # print('q,k,v pathified: ',q_g.shape, k_g.shape, v_g.shape)
            
            y, attn = self.attention(q_g, k_g, v_g)
            y = y.reshape(B, group_size, n_region, -1, C//self.num_heads)
            # print('attention out: ',y.shape)
            if i>0:
                repetition_factor = (N_g//y.shape[2])
                y = y.repeat(1, 1, repetition_factor, 1, 1)
                # print('attention out repeat: ',y.shape)
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