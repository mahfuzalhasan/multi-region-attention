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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, local_region_shape = [4, 8, 40]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
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

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
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

    """ arr.shape -> B x num_head x H x W x C """
    """ arr.shape -> B x num_head x H x W x C """
    # create overlapping patches
    def patchify(self, arr, H, W, patch_size):
        # print(arr.shape)
        unwrap = arr.view(arr.shape[0], arr.shape[1], H, W, arr.shape[3])
        B, Nh, H, W, Ch = unwrap.shape
        patches = unwrap.view(B, Nh, H // patch_size, patch_size, W // patch_size, patch_size, Ch)
        patches = patches.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, Nh, patch_size**2, Ch)
        return patches


    def attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale   # scaling needs to be fixed
        attn = attn.softmax(dim=-1)      #  couldn't figure out yet
        attn = self.attn_drop(attn)
        x = (attn @ v)
        return x


    def fuse_ms_attn_map(self, A, H, W):
        B, N, C = A[0].shape
        output_small_patched_attn = A[1].permute(0, 2, 1).contiguous().view(B, C, H, W)
        # output_small_patched_attn = self.attn_fusion[0](output_small_patched_attn, output_small_patched_attn)
        global_attn = A[0].permute(0, 2, 1).contiguous().view(B, C, H, W)
        ##print('shapes: ', output.shape, global_attn.shape)
        for i in range(2,len(A)):
            idx = i - 2
            output_small_patched_attn = self.attn_fusion[idx](output_small_patched_attn, A[i].permute(0, 2, 1).contiguous().view(B, C, H, W))
        output = self.global_fusion(output_small_patched_attn, global_attn)
        # print('final fused: ',output.shape)
        return output_small_patched_attn

    def forward(self, x, H, W):
        ####print('!!!!!!!!!!!!attention head: ',self.num_heads, ' !!!!!!!!!!')
        A = []
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # print(f' k:{k.shape} v:{v.shape} q:{q.shape}')
        # print('############################################')
        self.attn_outcome_per_head = []
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
                a_1 = self.attention(qh, kh, vh)
                # print('global attn: ',a_1.shape)
            else:
                q_patch = self.patchify(qh, H, W, rg_shp)
                k_patch = self.patchify(kh, H, W, rg_shp)
                v_patch = self.patchify(vh, H, W, rg_shp)

                # Here Nh = 1 as we are working on per head.
                # Grouping head will be experimented later
                B_, Nh, Np, Ch = q_patch.shape
                q_p, k_p, v_p = map(lambda t: rearrange(t, 'b h n d -> (b h) n d', h = Nh), (q_patch, k_patch, v_patch))
                patched_attn = self.attention(q_p, k_p, v_p)
                patched_attn = patched_attn.view(B_, Nh, Np, Ch)
                patched_attn = patched_attn.permute(0, 2, 1, 3).contiguous().reshape(B, N, Ch)
                a_1 = patched_attn.unsqueeze(dim=1)
                # print('final attn: ',a_1.shape)
            self.attn_outcome_per_head.append(a_1)

        #concatenating multi-scale outcome from different heads
        attn_fused = torch.cat(self.attn_outcome_per_head, axis=1)
        # print('attn_fused:',attn_fused.shape)
        attn_fused = attn_fused.permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        # print('fina attn_fused:',attn_fused.shape)
        attn_fused = self.proj(attn_fused)
        attn_fused = self.proj_drop(attn_fused )
        return attn_fused

if __name__=="__main__":
    # #######print(backbone)
    B = 4
    C = 3
    H = 32
    W = 32
    device = 'cuda:0'
    ms_attention = MultiScaleAttention(512, num_heads=2, sr_ratio=8, local_region_shape=[4, 8])
    ms_attention = ms_attention.to(device)
    # ms_attention = nn.DataParallel(ms_attention, device_ids = [0,1])
    # ms_attention.to(f'cuda:{ms_attention.device_ids[0]}', non_blocking=True)

    f = torch.randn(B, 1024, 512).to(device)
    ##print(f'input to multiScaleAttention:{f.shape}')
    y = ms_attention(f, 32, 32)
    ##print('output: ',y.shape)