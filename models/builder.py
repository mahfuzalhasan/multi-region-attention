import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
import numpy as np

from utils.logger import get_logger

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d, test=False):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.test = test
        self.input_size = (cfg.IMAGE.image_height, cfg.IMAGE.image_width)
        # print('Builder input: ',self.input_size)
        self.logger = get_logger()
        # import backbone and decoder
        if cfg.MODEL.backbone == 'mit_b0':
            self.logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]    # keep this must why???
            from .encoders.mra_transformer import mit_b0 as backbone
            self.backbone = backbone(fuse_cfg=cfg, norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b1':
            self.logger.info('Using backbone: Segformer-B1')
            from .encoders.mra_transformer import mit_b1 as backbone
            self.backbone = backbone(fuse_cfg=cfg, norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b2':
            self.logger.info('Using backbone: Segformer-B2')
            from .encoders.mra_transformer import mit_b2 as backbone
            self.backbone = backbone(fuse_cfg=cfg, norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b3':
            self.logger.info('Using backbone: Segformer-B3')
            from .encoders.mra_transformer import mit_b3 as backbone
            self.backbone = backbonebackbone(img_size=self.input_size, norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b4':
            self.logger.info('Using backbone: Segformer-B4')
            from .encoders.mra_transformer import mit_b4 as backbone
            self.backbone = backbonebackbone(img_size=self.input_size, norm_fuse=norm_layer)
        
        elif cfg.MODEL.backbone == 'mit_b5':
            self.logger.info('Using backbone: Segformer-B5')
            from .encoders.mra_transformer import mit_b5 as backbone
            self.backbone = backbone(img_size=self.input_size, norm_fuse=norm_layer)
        
        else:
            self.logger.error('Backbone not found!!! Currently only support mit_b0 - mit_b5')

        self.aux_head = None

        if cfg.MODEL.decoder == 'MLPDecoder':
            self.logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, 
                                        num_classes=cfg.DATASET.num_classes, 
                                        norm_layer=norm_layer, 
                                        embed_dim=cfg.MODEL.decoder_embed_dim)
        else:
            self.logger.error('Decoder not found!!! Currently only support MLPDecoder')
        

        self.criterion = criterion
        if self.criterion and not self.test:
            self.init_weights(cfg, pretrained=cfg.MODEL.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        self.logger.info('Initing weights ...')
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
