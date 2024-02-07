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
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.window_size = window_size
        self.n_local_region_scales = n_local_region_scales
        self.img_size = img_size

        self.n_local_heads = self.num_heads//self.n_local_region_scales     # heads per group
        self.H, self.W = img_size[0], img_size[1]

        self.N_G = self.H//self.window_size * self.W//self.window_size

        # assert self.num_heads%n_local_region_scales == 0
        # Linear embedding
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
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
        attn = attn.softmax(dim=-1)                     
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x, attn
    
    # need to inspect for learnable upsampling and concatenation
    def merge_attn_output(self, outputs):
        final = outputs[0]
        for i in range(1, len(outputs)):
            current = outputs[i]
            # print(f'current: {current.shape}')
            final = F.interpolate(final, size=current.size()[2:], mode='nearest')
            final = torch.cat([final, current], dim=1)
            # print(f'final: {final.shape}')
        return final


    def forward(self, x, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        # N = H*W
        self.H=H
        self.W=W
        A = []
        B, N, C = x.shape
        assert N==self.H*self.W

        group_size = int(math.ceil(self.num_heads/self.n_local_region_scales))
        
        # temp --> 3,B,N,C
        temp = self.qkv_proj(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).contiguous()
        self.attn_outcome_per_group = []
        self.attn_mat_per_head = []

        
        for i in range(self.n_local_region_scales):
            # print(f'$$$ group:{i} $$$$$$')
            local_C = C//self.n_local_region_scales     # channel per head-group, local_C = head_dim * heads_per_group(n_local_heads)
            qkv = temp[:, :, :, i*local_C:i*local_C + local_C]
            # 3*B, 56, 56, l_C
            qkv = qkv.reshape(-1, self.H, self.W, local_C).permute(0, 3, 1, 2).contiguous()
            
            ## pooling per group using adaptive avg pool. Need to inspect learnable downsampling
            pi = self.n_local_region_scales-i-1
            output_size = (self.H//int(pow(2,pi)), self.W//int(pow(2, pi)))
            if i<self.n_local_region_scales-1:
                qkv = F.adaptive_avg_pool2d(qkv, output_size)
            ############################################
                
            # 3*B, l_C, lH, lW
            B_, _, l_H, l_W = qkv.shape     # B_ = 3*B

            # Local attention on original patch resolution for a head-group
            if i==self.n_local_region_scales-1:
                qkv = qkv.view(B_, H // self.window_size, self.window_size, W // self.window_size, self.window_size, local_C)
                # B_, #num_l_reg_7x7, 49, l_C
                qkv = qkv.permute(0, 1, 3, 2, 4, 5).contiguous().view(B_, self.N_G, -1, local_C)
                # 3, B, local_heads, #num_l_reg_7x7, 49, head_dim
                qkv = qkv.reshape(3, B, self.N_G, -1, self.n_local_heads, self.head_dim).permute(0, 1, 4, 2, 3, 5).contiguous()
                # B, local_heads, #num_l_reg_7x7, 49, head_dim
                q, k, v = qkv[0], qkv[1], qkv[2]
                y, attn = self.attention(q, k, v)
                # print(f'y local: {y.shape}')
                y = y.permute(0, 1, 4, 2, 3).contiguous().reshape(B, local_C, N).view(B, local_C, self.H, self.W)

            else:  # global attention on reduced patch resolution for other head-groups
                # B_, local_heads, lH*lW, head_dim
                qkv = qkv.reshape(3, B, self.n_local_heads, self.head_dim, l_H*l_W).permute(0, 1, 2, 4, 3)
                # B, local_heads, lH*lW, head_dim
                q, k ,v = qkv[0], qkv[1], qkv[2]
                y, attn = self.attention(q, k, v)
                y = y.reshape(B, self.n_local_heads, l_H, l_W, self.head_dim).permute(0, 1, 4, 2, 3).contiguous().reshape(B, local_C, l_H, l_W)
            # print(f'y:{y.shape}')
            self.attn_outcome_per_group.append(y)

        attn_fused = self.merge_attn_output(self.attn_outcome_per_group)
        attn_fused = attn_fused.reshape(B, C, N).permute(0, 2, 1).contiguous()
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
    ms_attention = MultiScaleAttention(C, num_heads=24, n_local_region_scales=1, window_size=7, img_size=(7, 7))
    # ms_attention = ms_attention.to(device)
    # # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('output: ',y.shape)