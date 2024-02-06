import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowSEBlock(nn.Module):
    def __init__(self, num_windows, reduction=16):
        super(WindowSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Adapt for pooling across each window
        self.fc = nn.Sequential(
            nn.Linear(num_windows, num_windows // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_windows // reduction, num_windows, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: BxCxNxS, where N is number of windows, S is spatial dimension (flattened)
        b, c, n, s = x.shape
        # Reshape to treat each window equally and merge C and S for pooling
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(b, n, -1)  # Shape: BxNx(C*S)
        # Pooling across the combined C*S dimension to get BxNx1
        y = self.avg_pool(x_reshaped.view(b, n, c, s)).view(b, n)  # Now: BxN
        # Pass through FC to get recalibration weights
        y = self.fc(y).view(b, n, 1, 1)
        # Apply recalibration across windows
        recalibrated = x * y.expand(-1, -1, c, s).permute(0, 2, 1, 3)  # Expand and apply to original shape
        
        return recalibrated

class EnhanceWindowRelations(nn.Module):
    def __init__(self, channels, num_windows=64, region_size=49, reduction=16):
        super(EnhanceWindowRelations, self).__init__()
        self.window_se_block = WindowSEBlock(num_windows, reduction)

    def forward(self, x):
        # x is the attention output of shape BxCx64x49
        # Apply Window-based SE block to enhance relations among windows
        enhanced_x = self.window_se_block(x)
        return enhanced_x

# Example usage
B, C, num_windows, region_size = 10, 256, 64, 49  # Example dimensions
attention_output = torch.randn(B, C, num_windows, region_size)  # Example attention output

enhancer = EnhanceWindowRelations(C, num_windows, region_size)
enhanced_output = enhancer(attention_output)

print(enhanced_output.shape)  # Should be BxCx64x49, with inter-window relations enhanced

