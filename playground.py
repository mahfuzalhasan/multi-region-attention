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
        self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        # x1 = self.conv1(x)  # Reduces to (B, C, 28, 28)
        # print(f'step 1: ',x1.shape)
        # x2 = self.conv2(x)  # Reduces to (B, C, 14, 14)
        # print(f'step 2: ',x2.shape)
        # x3 = self.conv3(x)  # Reduces to (B, C, 7, 7)
        # print(f'step 3: ',x3.shape)
        x = self.conv_transpose(x)
        upsampled_x = F.interpolate(x, size=(7, 16*7), mode='nearest')
        print(f'up:{upsampled_x.shape}')
        return x

# Example usage
in_channels = 32*2  # Example input channels
model = DilatedConvReducer(in_channels)
B = 4
# Example input tensor of size (B, C, 14, 14)
input_tensor = torch.rand(B, in_channels, 7, 7*4)

# Apply the model to reduce dimensions
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Should print torch.Size([1, 3, 7, 7])
