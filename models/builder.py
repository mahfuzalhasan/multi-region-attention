import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
import numpy as np

# from engine.logger import get_logger

# logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d, test=False):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.test = test
        # import backbone and decoder
        if cfg.MODEL.backbone == 'swin_s':
            print('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.MODEL.backbone == 'swin_b':
            print('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b5':
            print('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b4':
            print('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b2':
            print('Using backbone: Segformer-B2')
            from .encoders.student_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b1':
            print('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.MODEL.backbone == 'mit_b0':
            print('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            print('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.MODEL.decoder == 'MLPDecoder':
            print('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.DATASET.num_classes, norm_layer=norm_layer, embed_dim=cfg.MODEL.decoder_embed_dim)
        
        elif cfg.MODEL.decoder == 'UPernet':
            print('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.DATASET.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.DATASET.num_classes, norm_layer=norm_layer)
        
        elif cfg.MODEL.decoder == 'deeplabv3+':
            print('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.DATASET.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.DATASET.num_classes, norm_layer=norm_layer)

        else:
            print('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.DATASET.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion and not self.test:
            self.init_weights(cfg, pretrained=cfg.MODEL.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            print('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        print('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.TRAIN.bn_eps, cfg.TRAIN.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.TRAIN.bn_eps, cfg.TRAIN.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb)
        else:
            out = self.encode_decode(rgb)
            
        if label is None:
            return out

        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss, out
