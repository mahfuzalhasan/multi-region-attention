import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import torch
import torch.nn as nn

# Define a custom module with dilated convolutions
import torch
import torch.nn.functional as F
import math

def merge_regions(x, merge_size):
    """
    Merge regions in the tensor x.
    x: Tensor of shape (B, H, R, N, C_h), where R is the number of regions.
    merge_size: Size of the block to merge (2 for 2x2, 4 for 4x4).
    """
    B, H, R, N, C_h = x.shape
    grid_size = int((R / merge_size**2) ** 0.5)  # New grid size after merge
    new_R = grid_size ** 2  # Number of regions after merge
    r = int(math.sqrt(R))
    # Reshape to treat the regions in a 2D grid format and merge
    x_reshaped = x.view(B, H, r, r, N, C_h)
    x_merged = F.avg_pool3d(x_reshaped.permute(0, 1, 4, 2, 3, 5).reshape(B, H, N, r, r * C_h),
                            kernel_size=(1, merge_size, merge_size),
                            stride=(1, merge_size, merge_size)).reshape(B, H, N, grid_size, grid_size, C_h)
    x_merged_2 = F.max_pool3d(x_reshaped.permute(0, 1, 4, 2, 3, 5).reshape(B, H, N, r, r * C_h),
                            kernel_size=(1, merge_size, merge_size),
                            stride=(1, merge_size, merge_size)).reshape(B, H, N, grid_size, grid_size, C_h)

    # Reshape back to the original format, but now with new_R regions
    x_merged = x_merged.permute(0, 1, 3, 4, 2, 5).reshape(B, H, new_R, N, C_h)
    x_merged_2 = x_merged_2.permute(0, 1, 3, 4, 2, 5).reshape(B, H, new_R, N, C_h)
    print(x_merged.shape, x_merged_2.shape)
    return x_merged+x_merged_2



def merge_regions_spatial(x, merge_size):
    # x shape is expected to be (B, H, R, N, C_h)
    B, H, R, N, C_h = x.shape
    
    # Determine the new grid size based on the merge size
    grid_size = int((R // (merge_size ** 2)) ** 0.5)
    new_R = grid_size ** 2  # Number of regions after merge
    r = int(math.sqrt(R))
    # Reshape x to treat regions in a 2D grid format and separate channels
    x_reshaped = x.view(B, H, r, r, N, C_h)
    
    # Apply pooling over the spatial dimensions representing the regions
    # while preserving the separate channel information
    x_pooled = F.avg_pool3d(x_reshaped.permute(0, 1, 5, 2, 3, 4).reshape(B, H * C_h, r, r, N),
                            kernel_size=(merge_size, merge_size, 1),
                            stride=(merge_size, merge_size, 1)).reshape(B, H, C_h, grid_size, grid_size, N)
    
    # Reshape back to match the expected output format
    x_merged = x_pooled.permute(0, 1, 3, 4, 5, 2).reshape(B, H, new_R, N, C_h)
    
    return x_merged

# Example tensor
B, H, R, N, C_h = 8, 4, 64, 49, 32
attention_tensor = torch.randn(B, H, R, N, C_h)

# Split the tensor into 3 groups along the head dimension
# group_1 = attention_tensor[:, :2, :, :, :]
# group_2 = attention_tensor[:, 2:4, :, :, :]
# group_3 = attention_tensor[:, 4:, :, :, :]

group_1 = attention_tensor[:, 0, :, :, :]
group_2 = attention_tensor[:, 0:1, :, :, :]
group_3 = attention_tensor[:, 1:2, :, :, :]
group_4 = attention_tensor[:, 2:3, :, :, :]
print(group_1.shape, group_2.shape)
# Merge regions for group_2 (2x2 merge) and group_3 (4x4 merge)
group_2_merged = merge_regions_spatial(group_2, merge_size=2)
group_3_merged = merge_regions_spatial(group_3, merge_size=4)
group_4_merged = merge_regions_spatial(group_4, merge_size=8)

print("Group 1 shape:", group_1.shape)  # Expected: (B, 2, 16, 49, 32)
print("Group 2 merged shape:", group_2_merged.shape)  # Expected: (B, 2, 4, 49, 32)
print("Group 3 merged shape:", group_3_merged.shape)  # Expected: (B, 2, 1, 49, 32)
print("Group 4 merged shape:", group_4_merged.shape)  # Expected: (B, 2, 1, 49, 32)
