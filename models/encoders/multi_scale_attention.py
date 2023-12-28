import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from einops import rearrange, repeat
import math
import time
import copy


# RegionVIT: https://github.com/IBM/RegionViT/tree/master
# patch_tokens: (BxH/rxW/r)x(rxr)xC
def convert_to_spatial_layout(patch_tokens, output_channels, B, new_H, new_W, region_size, mask, p_l, p_r, p_t, p_b):
    """
    Convert the token layer from flatten into 2-D, will be used to downsample the spatial dimension.
    """
    # patch_tokens: (BxH/sxW/s)x(ksxks)xC

    Ch = output_channels
    # reorganize data, need to convert back to patch_tokens: BxCxHxW
    patch_tokens = patch_tokens.transpose(1, 2).reshape((B, -1, region_size * region_size* Ch)).transpose(1, 2)
    patch_tokens = F.fold(patch_tokens, (new_H, new_W), kernel_size=region_size, stride=region_size, padding=(0, 0))

    if mask is not None:
        if p_b > 0:
            patch_tokens = patch_tokens[:, :, :-p_b, :]
        if p_r > 0:
            patch_tokens = patch_tokens[:, :, :, :-p_r]

    return patch_tokens


# RegionVIT: https://github.com/IBM/RegionViT/tree/master
def convert_to_flatten_layout(patch_tokens, ws):
    # padding if needed, and all paddings are happened at bottom and right.
    B, C, H, W = patch_tokens.shape
    
    need_mask = False
    
    H_ks = math.ceil(H/ws)
    W_ks = math.ceil(W/ws)
    p_l, p_r, p_t, p_b = 0, 0, 0, 0
    if H % ws != 0 or W % ws != 0 and ws!=1:
        p_l, p_r = 0, W_ks * ws - W
        p_t, p_b = 0, H_ks * ws - H
        patch_tokens = F.pad(patch_tokens, (p_l, p_r, p_t, p_b))
        need_mask = True
    
    B, C, H, W = patch_tokens.shape
    kernel_size = (H // H_ks, W // W_ks)
    tmp = F.unfold(patch_tokens, kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))  # Nx(Cxksxks)x(H/sxK/s)
    patch_tokens = tmp.transpose(1, 2).reshape(-1, C, kernel_size[0] * kernel_size[1]).transpose(-2, -1)  # (NxH/sxK/s)x(ksxks)xC
    if need_mask:
        BH_sK_s, ksks, C = patch_tokens.shape
        H_s, W_s = H // ws, W // ws
        mask = torch.ones(BH_sK_s // B, ksks, ksks, device=patch_tokens.device, dtype=torch.float)
        
        right = torch.zeros(ksks, ksks, device=patch_tokens.device, dtype=torch.float)
        tmp = torch.zeros(ws, ws, device=patch_tokens.device, dtype=torch.float)
        tmp[0:(ws - p_r), 0:(ws - p_r)] = 1.
        right = tmp.repeat(ws, ws)
        
        bottom = torch.zeros_like(right)
        bottom[0:ws*(ws - p_b), 0:ws*(ws - p_b)] = 1.
        
        bottom_right = torch.zeros_like(right)
        bottom_right[0:ws*(ws-p_b), 0:ws*(ws-p_r)] =  right[0:ws*(ws-p_b), 0:ws*(ws-p_r)]

        mask[W_s - 1:(H_s - 1) * W_s:W_s, ...] = right
        mask[(H_s - 1) * W_s:, ...] = bottom
        mask[-1, ...] = bottom_right
        mask = mask.repeat(B, 1, 1) 
    else:
        mask = None

    return patch_tokens, mask, p_l, p_r, p_t, p_b, B, C, H, W



class MultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                    proj_drop=0., sr_ratio=1, local_region_shape = [4, 8, 40], img_size=(32,32)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.img_size = img_size
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.local_region_shape = local_region_shape
        assert len(local_region_shape)==self.num_heads
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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

    
    """ arr.shape: B, N, Ch"""
    # create overlapping patches
    def patchify(self, arr, H, W, region):
        unwrap = arr.view(arr.shape[0], H, W, arr.shape[2]).permute(0, 3, 1, 2) #B,C,H,W
        flatten, mask, p_l, p_r, p_t, p_b, B, Ch, H, W = convert_to_flatten_layout(unwrap, region)
        return flatten, mask, p_l, p_r, p_t, p_b, B, Ch, H, W


    def attention(self, corr, v):
        attn = corr.softmax(dim=-1)      
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x, attn

    def attention_global(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x

    def forward(self, x, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        # N = H*W
        self.H=H
        self.W=W
        A = []
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        self.attn_outcome_per_head = []
        self.correlation_matrices = []
        for i in range(self.num_heads):
            qh = q[:, i, :, :]
            kh = k[:, i, :, :]
            vh = v[:, i, :, :]
            region = self.local_region_shape[i]
            if region == 1:
                a_1 = self.attention_global(qh, kh, vh)
                a_1 = a_1.unsqueeze(dim=1)  # to introduce head dimension
            else:
                # patch: B_Hr_Wr x Np x Ch, mask:B x (Np) x (Np)
                q_patch, mask, p_l, p_r, p_t, p_b, B, Ch, new_H, new_W = self.patchify(qh, H, W, region)
                k_patch, mask, _, _, _, _, _, _, _, _ = self.patchify(kh, H, W, region)
                v_patch, mask, _, _, _, _, _, _, _, _ = self.patchify(vh, H, W, region)

                B_Hr_Wr, Np, Ch = q_patch.shape
                
                # B_r_r, Np, Np    where Np = region^2, for whole image Np=N
                correlation = (q_patch @ k_patch.transpose(-2, -1)) * self.scale
                if mask is not None:
                    correlation = correlation.masked_fill(mask == 0, torch.finfo(correlation.dtype).min)
                
                # (B_Hr_Wr, Np, Ch), (B_Hr_Wr, Np, Np)
                patched_attn, attn_matrix = self.attention(correlation, v_patch)
                
                patched_attn = convert_to_spatial_layout(patched_attn, Ch, B, new_H, new_W, region, mask, p_l, p_r, p_t, p_b)
                patched_attn = patched_attn.reshape(B, Ch, N).permute(0, 2, 1)
                a_1 = patched_attn.unsqueeze(dim=1) # To introduce head dimension

            self.attn_outcome_per_head.append(a_1)

        #concatenating multi-scale-region attention outcome from different heads
        # B, head, N, Ch
        attn_fused = torch.cat(self.attn_outcome_per_head, axis=1)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused)
        
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
    C = 96
    H = 14
    W = 14
    device = 'cuda:1'
    ms_attention = MultiScaleAttention(C, num_heads=4, sr_ratio=8, 
                                local_region_shape=[3, 3, 7, 7], img_size=(14,14))
    ms_attention = ms_attention.to(device)
    # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, H*W, C).to(device)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, H, W)
    print('output: ',y.shape)