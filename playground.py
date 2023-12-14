import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy

B, C, H, W = 1, 1, 7, 7
patch_tokens = torch.randn(B,C,H,W)
ws = 3



B, C, H, W = patch_tokens.shape
    
need_mask = False

H_ks = math.ceil(H/ws)
W_ks = math.ceil(W/ws)
print(f"H_ks{H_ks} W_ks:{W_ks}")
p_l, p_r, p_t, p_b = 0, 0, 0, 0
if (H % ws != 0 or W % ws != 0) and ws!=1:
    p_l, p_r = 0, W_ks * ws - W
    p_t, p_b = 0, H_ks * ws - H
    patch_tokens = F.pad(patch_tokens, (p_l, p_r, p_t, p_b))
    need_mask = True
    print(f'patch token padded: {patch_tokens.shape}')
    print(f'pads: pl:{p_l}, pr:{p_r}, pt:{p_t}, pb:{p_b}')

B, C, H, W = patch_tokens.shape
kernel_size = (H // H_ks, W // W_ks)
print(f'kernel: {kernel_size}')
tmp = F.unfold(patch_tokens, kernel_size=kernel_size, stride=kernel_size, padding=(0, 0))  # Nx(Cxksxks)x(H/sxK/s)
patch_tokens = tmp.transpose(1, 2).reshape(-1, C, kernel_size[0] * kernel_size[1]).transpose(-2, -1)  # (NxH/sxK/s)x(ksxks)xC
print(f'patched tokens:{patch_tokens.shape}')
if need_mask:
    BH_sK_s, ksks, C = patch_tokens.shape
    H_s, W_s = H // ws, W // ws
    mask = torch.ones(BH_sK_s // B, ksks, ksks, device=patch_tokens.device, dtype=torch.float)
    print(f'created mask: {mask.shape}')
    right = torch.zeros(ksks, ksks, device=patch_tokens.device, dtype=torch.float)
    tmp = torch.zeros(ws, ws, device=patch_tokens.device, dtype=torch.float)
    print(f'tmp: \n {tmp} shape: \n {tmp.shape}')
    tmp[0:(ws - p_r), 0:(ws - p_r)] = 1.
    print(f'tmp after fixation \n {tmp}')
    tmp = tmp.repeat(ws, ws)
    print(f'tmp after repeat: \n {tmp.shape} \n tmp:\n{tmp}')
    # right[1:, 1:] = tmp
    right = tmp
    # right[0, 0] = 1
    # right[0, 1:] = torch.tensor([1.] * (ws - p_r) + [0.] * p_r).repeat(ws).to(right.device)
    # right[1:, 0] = torch.tensor([1.] * (ws - p_r) + [0.] * p_r).repeat(ws).to(right.device)
    print(f'right:{right} shape:{right.shape}')
    bottom = torch.zeros_like(right)
    bottom[0:ws * (ws - p_b), 0:ws * (ws - p_b)] = 1

    bottom_right = torch.zeros_like(right)
    print(f'br shape:{bottom_right.shape} {ws*(ws-p_b)}')
    print(bottom_right)
    bottom_right[0:ws*(ws-p_b), 0:ws*(ws-p_r)] =  right[0:ws*(ws-p_b), 0:ws*(ws-p_r)]
    print(f'br:{bottom_right} shape:{bottom_right.shape}')

    mask[W_s - 1:(H_s - 1) * W_s:W_s, ...] = right
    mask[(H_s - 1) * W_s:, ...] = bottom
    mask[-1, ...] = bottom_right
    # print(f'final mask[0]: \n {mask[0]} shape:{mask.shape}')
    # print(f'final mask[1]: \n {mask[1]} shape:{mask.shape}')
    # print(f'final mask[2]: \n {mask[2]} shape:{mask.shape}')
    mask = mask.repeat(B, 1, 1)
    print(f'final mask:\n {mask} \n shape:{mask.shape}')
    
else:
    mask = None
