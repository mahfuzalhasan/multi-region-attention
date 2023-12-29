import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalSubsampledAttention(nn.Module):
    def __init__(self, dim, num_heads, window_shape):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.window = window_shape

    # create overlapping patches
    def patchify(self, arr, H, W, patch_size):
        B, Nh, N, C = arr.shape
        unwrap = arr.view(B, Nh, H, W, arr.shape[3])
        B, Nh, H, W, Ch = unwrap.shape
        patches = unwrap.view(B, Nh, H // patch_size, patch_size, W // patch_size, patch_size, Ch)
        patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, Nh, -1, patch_size**2, Ch)
        return patches
    # B x Nh x (H/pxW/p) x p^2 x C
    def representative(self, x):
        B, Nh, N_region, p_vec, C = x.shape
        x = x.permute(0, 1, 2, 4, 3).contiguous().reshape(-1, C, self.window, self.window)
        x = self.pool(x)
        x = x.reshape(B, Nh, N_region, C).unsqueeze(3)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Subsampling
        # BxNhx(H/rxW/r)xr**2XC
        q = self.patchify(q, H, W, self.window)
        k = self.patchify(k, H, W, self.window)
        v = self.patchify(v, H, W, self.window)
        # print('q patched: ',q.shape)
        k = self.representative(k)
        v = self.representative(v)
        # print('k,v mean: ',k.shape, v.shape)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)

if __name__=="__main__":
    H = 56
    W = 56
    C = 96
    B = 8
    x = torch.randn(B, H, W, C)
    x = x.reshape(B, -1, C)
    gsa = GlobalSubsampledAttention(C, 3, 14)
    
    y = gsa(x, H, W)
    print(y.shape)