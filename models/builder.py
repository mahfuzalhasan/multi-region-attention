import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from functools import partial
import numpy as np


from configs.config_imagenet import config as cfg

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from utils.logger import get_logger

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.LayerNorm, test=False):
        super(EncoderDecoder, self).__init__()
        self.channels = cfg.MODEL.decoder_embed_dim
        self.norm_layer = norm_layer
        self.test = test
        self.input_size = (cfg.IMAGE.image_height, cfg.IMAGE.image_width)
        # print('Builder input: ',self.input_size)
        self.logger = get_logger()
        # import backbone and decoder
        if cfg.MODEL.backbone == 'mra_tiny':
            self.logger.info('Using backbone: Segformer-B0')
            from .encoders.mra_transformer import mit_b0 as backbone
            self.backbone = backbone(fuse_cfg=cfg, norm_layer = norm_layer)
        
        elif cfg.MODEL.backbone == 'mra_small':
            self.logger.info('Using backbone: Segformer-B1')
            from .encoders.mra_transformer import mit_b1 as backbone
            self.backbone = backbone(fuse_cfg=cfg, norm_layer=norm_layer)
        
        elif cfg.MODEL.backbone == 'mra_base':
            self.logger.info('Using backbone: Segformer-B2')
            from .encoders.mra_transformer import mit_b2 as backbone
            self.backbone = backbone(fuse_cfg=cfg, norm_layer=norm_layer)
        else:
            self.logger.error('Backbone not found!!! Currently only support mit_b0 - mit_b5')

        self.aux_head = None

        if cfg.MODEL.decoder == 'ClassificationHead':
            self.logger.info('Using Classification Head')
            from decoders.classifier import Classifier
            self.decode_head = Classifier(in_channels=self.channels, 
                                        num_classes=cfg.DATASET.NUM_CLASSES)
        else:
            self.logger.error('Decoder not found!!! Currently only support MLPDecoder')
        

        self.criterion = criterion
        # Not necessary during imagenet training.
        # Weight gets initialized inside encoder and classification head.
        
        # if self.criterion and not self.test:
        #     self.init_weights(cfg, pretrained=cfg.MODEL.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        self.logger.info('Initiating weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.TRAIN.bn_eps, cfg.TRAIN.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb)
        out = self.decode_head.forward(x)
        return out

    def forward(self, rgb, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb)
        else:
            out = self.encode_decode(rgb)
        # print(f'out:{out.shape}')
        if label is None:
            return out

        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            # print(f'from builder out:{out.size()} loss:{loss}')
            return loss, out

if __name__=="__main__":
    criterion = None
    model=EncoderDecoder(cfg=cfg, criterion=criterion, norm_layer=nn.BatchNorm2d)
    B = 4
    C = 3
    H = 224
    W = 224
    device = 'cuda:1'
    rgb = torch.randn(B, C, H, W)
    y = model(rgb)
    print("final output: ", y.shape)
