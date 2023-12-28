from torch.nn.modules import module
import torch.nn.functional as F
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Classifier(nn.Module):
    def __init__(self, in_channels=384, num_classes=40):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.head(x)
        return output