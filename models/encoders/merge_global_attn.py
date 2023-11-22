import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import math
import time

def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
    """Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

# ms_attention = MultiScaleAttention(512, num_heads=8, sr_ratio=8, local_region_shape=[2, 2, 2, 2, 1, 1, 1, 1])

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
        #print(f'local region shape:{self.local_region_shape}')
        assert len(local_region_shape)==self.num_heads
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # # Linear embedding for cross attention
        # unique_vals = sorted(list(set(self.local_region_shape)))
        # if unique_vals.count(1)>0:
        #     unique_vals.remove(1)
        # self.unique_vals = unique_vals

        # self.q_cross = nn.ModuleList()
        # self.kv_cross = nn.ModuleList()
        # # self.v_cross = nn.ModuleList()
        # for _ in range(len(self.unique_vals)-1):
        #     self.q_cross.append(nn.Linear(head_dim, head_dim))
        #     self.kv_cross.append(nn.Linear(head_dim, head_dim*2))
        #     # self.v_cross.append(nn.Linear(head_dim, head_dim))


    
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

    def calc_index(self, region_size):
        index = self.unique_vals.index(region_size)
        return index

    """ arr.shape -> B x num_head x H x W x C """
    """ arr.shape -> B x num_head x H x W x C """
    # create overlapping patches
    def patchify(self, arr, H, W, patch_size):
        unwrap = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
        B, Nh, H, W, Ch = unwrap.shape
        patches = unwrap.view(B, Nh, H // patch_size, patch_size, W // patch_size, patch_size, Ch)
        patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, Nh, -1, patch_size**2, Ch)
        return patches


    def attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)     
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x, attn
    



    def forward(self, x, H, W):
        A = []
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn_outcome_per_head = []
        global_attn = []
        # print('region shapes: ',self.local_region_shape)
        for i in range(self.num_heads):
            qh = q[:, i, :, :]
            qh = torch.unsqueeze(qh, dim=1)
            kh = k[:, i, :, :]
            kh = torch.unsqueeze(kh, dim=1)
            vh = v[:, i, :, :]
            vh = torch.unsqueeze(vh, dim=1)
            
            rg_shp = self.local_region_shape[i]

            # B, Nh, N_patch, Np, C
            q_patch = self.patchify(qh, H, W, rg_shp)
            k_patch = self.patchify(kh, H, W, rg_shp)
            v_patch = self.patchify(vh, H, W, rg_shp)
            B, Nh, N_Patch, Np, Ch = q_patch.shape
            # (B, Nh, N_patch, Np, Ch), (B, Nh, N_patch, Np, Np)
            patched_attn, attn_matrix = self.attention(q_patch, k_patch, v_patch)
            
            a_1 = patched_attn.reshape(B, N, Ch)
            a_1 = a_1.unsqueeze(dim=1)     # per head --> 1 dimension
           
            # B, N_patch, Np, Ch --> B, Np, N_patch, Ch --> B, Np, C 
            gl_sub = patched_attn.squeeze(dim=1).transpose(1,2).mean(dim=2).view(B, rg_shp, rg_shp, Ch).permute(0, 3, 1, 2)
            # print(gl_sub.shape)
            if len(global_attn)>0:
                gl_sub_previous = global_attn[-1]
                r = gl_sub.shape[2]//gl_sub_previous.shape[2]
                if r>1:
                    hl, wl = gl_sub_previous.shape[2], gl_sub_previous.shape[3] 
                    gl_sub_previous = F.interpolate(gl_sub_previous, [hl*r, wl*r], mode='bilinear')

                gl_sub += gl_sub_previous

            global_attn.append(gl_sub)

            if i==self.num_heads-1:         #This is last layar and largest head size
                c_global_sub_feat = global_attn[-1].reshape(B, -1, Ch).unsqueeze(dim=1).unsqueeze(dim=1)
                # print(f'\n $$$ final c_global_sub_feat: {c_global_sub_feat.shape} \n $$')
                # print(f'\n ^^^^^ k patch: {k_patch.shape} \n ^^^^^^')
                # print(f'\n ^^^^^ a_1: {a_1.shape} \n ^^^^^^')
                global_sub_attn, _ = self.attention(c_global_sub_feat, k_patch, v_patch)
                # print(f'\n #### global_subsampled attn: {global_sub_attn.shape} \n ###')
                global_sub_attn = global_sub_attn.reshape(B, 1, -1, Ch)
                a_1+=global_sub_attn

            attn_outcome_per_head.append(a_1)

        #concatenating multi-scale outcome from different heads
        attn_fused = torch.cat(attn_outcome_per_head, axis=1)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused )
        return attn_fused

if __name__=="__main__":
    # ########print(backbone)
    B = 4
    C = 3
    H = 480
    W = 640
    device = 'cuda:1'
    ms_attention = MultiScaleAttention(96, num_heads=4, sr_ratio=8, 
                                local_region_shape=[1, 4, 8, 16, 16], img_size=(128,128))
    ms_attention = ms_attention.to(device)
    # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, 16384, 96).to(device)
    ###print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, 128, 128)
    ###print('output: ',y.shape)