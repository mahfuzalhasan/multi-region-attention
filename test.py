import os.path as osp
import os
import sys
import time
import argparse
# import tqdm
from datetime import datetime
import numpy as np

import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

## Dataset-specific imports
# from config import config
from config_cityscapes import config
from eval import SegEvaluator
# from dataloader.dataloader import get_train_loader

from models.builder import EncoderDecoder as segmodel
# from dataloader.RGBXDataset import RGBXDataset


from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.metric import cal_mean_iou

from tensorboardX import SummaryWriter


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from dataloader.cfg_defaults import get_cfg_defaults
from dataloader.cityscapes_dataloader import CityscapesDataset
from val_segformer_rgbonly import val_cityscape


def Main(parser, cfg, args):
    with Engine(custom_parser=parser) as engine:
        
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, test=True)
        model = nn.DataParallel(model, device_ids = config.device_ids)
        model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)


        
        # <----------------- load model ----------------->
        saved_model_path = os.path.join(config.checkpoint_dir, 'model_495_multiscalemultihead.pth')
        print('model: ',saved_model_path)
        # exit()
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict['model'], strict=True)
        print(f'model loaded')
        epoch = state_dict['epoch']
        
        # model.eval()
        
        cityscapes_test = CityscapesDataset(cfg, split='val')
        test_loader = DataLoader(cityscapes_test, batch_size=1, shuffle=False, num_workers=4) # batchsize?
        print(f'total test sample: {len(cityscapes_test)} v_iteration:{len(test_loader)}')
        
        val_loss, val_mean_iou = val_cityscape(epoch, test_loader, model)
        
if __name__ == '__main__':
    print('entering main')
    parser = argparse.ArgumentParser(description="Test cityscapes Loader")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    #parser.add_argument('devices', default='0,1', type=str)
    args = parser.parse_args()
    print(args.opts)
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file) # dataloader/cityscapes_rgbd_config.yaml
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    logger = get_logger()

    os.environ['MASTER_PORT'] = '169710'
    
    Main(parser, cfg, args)    