import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import torch
import torch.nn as nn

# Define a custom module with dilated convolutions
class DilatedConvReducer(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConvReducer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, dilation=1, groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=4, padding=0, dilation=3, groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=8, padding=0, dilation=7, groups=in_channels)

    def forward(self, x):
        x1 = self.conv1(x)  # Reduces to (B, C, 28, 28)
        print(f'step 1: ',x1.shape)
        x2 = self.conv2(x)  # Reduces to (B, C, 14, 14)
        print(f'step 2: ',x2.shape)
        x3 = self.conv3(x)  # Reduces to (B, C, 7, 7)
        print(f'step 3: ',x3.shape)
        return x1, x2, x3

# Example usage
in_channels = 3  # Example input channels
model = DilatedConvReducer(in_channels)

# Example input tensor of size (B, C, 14, 14)
input_tensor = torch.rand(1, in_channels, 14, 14)

# Apply the model to reduce dimensions
output_tensor = model(input_tensor)
# print(output_tensor.shape)  # Should print torch.Size([1, 3, 7, 7])
