import torch
import torch.nn as nn

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
        assert len(local_region_shape)==self.num_heads
        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        unique_vals = sorted(list(set(self.local_region_shape)))
        if unique_vals.count(1)>0:
            unique_vals.remove(1)

        corr_projections = []
        
        for i in range(len(unique_vals)-1):
            
            small_patch = unique_vals[i]    # 4
            large_patch = unique_vals[i+1] # 8

            #print(small_patch, large_patch)

            in_channel, out_channel = self.proj_channel_conv(small_patch, large_patch)

            c_p = nn.Conv2d(in_channel, out_channel, 1)

            corr_projections.append(c_p)

        self.corr_projections = nn.ModuleList(corr_projections)

        #print('corr_proj convs: ',self.corr_projections)
            
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def proj_channel_conv(self, small_patch, large_patch):
        N = self.img_size[0] * self.img_size[1]   # (1024 = 32 x 32)

        N_small_patch = N // (small_patch ** 2)     # 64
        N_large_patch = N // (large_patch ** 2)     # 16

        #print('Ns, Nl: ',N_small_patch, N_large_patch)
        ratio = (large_patch ** 2) // (small_patch ** 2)    # 4

        #print('ratio: ',ratio)

        # sa = (B, 1, 64, 16 ,16)
        # ba = (B, 1, 16, 64 ,64)

        # sa --> ba : sa =  (B, 1, 64/(4**2), 64, 64) = (B, 1, 4, 64, 64)

        # bsa = sa concat ba = B, 1, 20, 64, 64
        #  bsa --> ba = 1x1conv(( 4 + 16), 16, 1)

        reduced_patch = N_small_patch // (ratio**2)   

        #print('red: ',reduced_patch)  
        
        in_channel = reduced_patch + N_large_patch
        return in_channel, N_large_patch

    def calc_index(self, patch_size):
        unique_vals = sorted(list(set(self.local_region_shape)))
        if unique_vals.count(1)>0:
            unique_vals.remove(1)
        index = unique_vals.index(patch_size)
        return index


    
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

    """ arr.shape -> B x num_head x H x W x C """
    """ arr.shape -> B x num_head x H x W x C """
    # create overlapping patches
    def patchify(self, arr, H, W, patch_size):
        # #print(arr.shape)
        unwrap = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
        B, Nh, H, W, Ch = unwrap.shape
        patches = unwrap.view(B, Nh, H // patch_size, patch_size, W // patch_size, patch_size, Ch)
        patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, Nh, -1, patch_size**2, Ch)
        return patches


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

    # correlation --> B, nh, N_patch, Np, Np
    def merge_correlation_matrices(self, correlation, head_idx):

        if self.local_region_shape[head_idx-1]==self.local_region_shape[head_idx]:
            # SILU
            correlation += self.correlation_matrices[-1]
        else:
            
            small_corr_matrix = self.correlation_matrices[-1]   #B,1,64,16,16
            #print(f'small corr matrices:{small_corr_matrix.shape} ')
            B, nh, N_patch_s, Np_s, Np_s = small_corr_matrix.shape
            _, _, _, Np_l, Np_l = correlation.shape             #B,1,16,64,64
            #print(f'large corr matrices:{correlation.shape} ')
            small_corr_matrix = small_corr_matrix.view(B, nh, -1, Np_l, Np_l) #B,1,4,64,64
            #print(f'reshape small corr matrices:{small_corr_matrix.shape} ')
            correlation = torch.cat([correlation, small_corr_matrix],axis=2)#B,1,20,64,64
            correlation = correlation.squeeze(dim=1)    #B,20,64,64
            #print(f'concat both:{correlation.shape} ')
            
            index = self.calc_index(self.local_region_shape[head_idx-1])
            #print(f' index: {index}, layer:{self.corr_projections[index]}')
            correlation = self.corr_projections[index](correlation)#B,16,64,64
            correlation = correlation.unsqueeze(dim=1)  #B,1,16,64,64

        return correlation


    def forward(self, x, H, W):
        #####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        A = []
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # print(f' k:{k.shape} v:{v.shape} q:{q.shape}')
        # print('############################################')
        self.attn_outcome_per_head = []
        self.correlation_matrices = []
        for i in range(self.num_heads):
            qh = q[:, i, :, :]
            qh = torch.unsqueeze(qh, dim=1)
            kh = k[:, i, :, :]
            kh = torch.unsqueeze(kh, dim=1)
            vh = v[:, i, :, :]
            vh = torch.unsqueeze(vh, dim=1)
            # print(f' head-wise k:{kh.shape} v:{vh.shape} q:{qh.shape}')
            rg_shp = self.local_region_shape[i]
            if rg_shp == 1:
                a_1 = self.attention_global(qh, kh, vh)
                # print('global attn: ',a_1.shape)
            else:
                # B, Nh, N_patch, Np, C
                q_patch = self.patchify(qh, H, W, rg_shp)
                k_patch = self.patchify(kh, H, W, rg_shp)
                v_patch = self.patchify(vh, H, W, rg_shp)

                
                B, Nh, N_Patch, Np, Ch = q_patch.shape
                # q_p, k_p, v_p = map(lambda t: rearrange(t, 'b h n d -> (b h) n d', h = Nh), (q_patch, k_patch, v_patch))
                
                # B, Nh, N_patch, Np, Np    where Np = p^2, for whole image Np=N
                correlation = (q_patch @ k_patch.transpose(-2, -1)) * self.scale
                if len(self.correlation_matrices)>0:
                    correlation = self.merge_correlation_matrices(correlation, i)
                self.correlation_matrices.append(correlation)
                
                # (B, Nh, N_patch, Np, C), (B, Nh, N_patch, Np, Np)
                patched_attn, attn_matrix = self.attention(correlation, v_patch)
                #print(f'pa shape: {patched_attn.shape}')
                # patched_attn = patched_attn.view(B, Nh, N_patch, Ch)
                patched_attn = patched_attn.reshape(B, N, Ch)
                a_1 = patched_attn.unsqueeze(dim=1)
                #print('final attn: ',a_1.shape)
            self.attn_outcome_per_head.append(a_1)

        #concatenating multi-scale outcome from different heads
        attn_fused = torch.cat(self.attn_outcome_per_head, axis=1)
        #print('attn_fused:',attn_fused.shape)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        #print('fina attn_fused:',attn_fused.shape)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused )
        return attn_fused

if __name__=="__main__":
    # #######print(backbone)
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
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, 128, 128)
    ##print('output: ',y.shape)