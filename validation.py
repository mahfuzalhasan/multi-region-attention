import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
import cv2

import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

import torchvision
from torchvision import transforms
from torchvision.utils import save_image

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
from utils.metric import cal_mean_iou, hist_info, compute_score

from tensorboardX import SummaryWriter


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_metric(results):
    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    count = 0
    for d in results:
        hist += d['hist']
        correct += d['correct']
        labeled += d['labeled']
        count += 1
    iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    print(f'iou per class:{iou} miou:{mean_IoU}')
    result_dict = dict(mean_iou=mean_IoU, freq_iou=freq_IoU, mean_pixel_acc=mean_pixel_acc)
    return result_dict

def val_cityscape(epoch, val_loader, model):
    model.eval()
    sum_loss = 0
    m_iou_batches = []
    all_results = []
    unique_values = []
    # path = './output_check'

    # invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
    #                                                  std = [ 1/0.190, 1/0.190, 1/0.185 ]),
    #                             transforms.Normalize(mean = [ -291, -0.329, -0.291 ],
    #                                                  std = [ 1., 1., 1. ]),
    #                            ])
    # data_mean = np.asarray([0.291,  0.329,  0.291])
    # data_std = np.asarray([0.190,  0.190,  0.185])
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            imgs = sample['image']      #B, 3, 1024, 2048
            gts = sample['label']       #B, 1024, 2048
            
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

            imgs_1, imgs_2 = imgs[:, :, :, :1024], imgs[:, :, :, 1024:]
            gts_1, gts_2 = gts[:, :, :1024], gts[:, :, 1024:]
            loss_1, out_1 = model(imgs_1, gts_1)
            loss_2, out_2 = model(imgs_2, gts_2)

            out = torch.cat((out_1, out_2), dim = 3)
            # mean over multi-gpu result
            loss = torch.mean(loss_1) + torch.mean(loss_2)
            #miou using torchmetric library
            # m_iou = cal_mean_iou(out, gts)

            score = out[0]      #1, C, H, W --> C, H, W = 19, H, W
            score = torch.exp(score)    
            score = score.permute(1, 2, 0)  #H,W,C
            pred = score.argmax(2)  #H,W
            
            pred = pred.detach().cpu().numpy()
            gts = gts[0].detach().cpu().numpy() #1, H, W --> H, W
            confusionMatrix, labeled, correct = hist_info(config.num_classes, pred, gts)
            results_dict = {'hist': confusionMatrix, 'labeled': labeled, 'correct': correct}
            all_results.append(results_dict)
            # if idx==2:
            #     print(all_results)
            #     # exit()

            # m_iou_batches.append(m_iou)

            sum_loss += loss

            # print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
            #         + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
            #         + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.val_print_stats == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'sample {idx}')

        val_loss = sum_loss/len(val_loader)
        result_dict = compute_metric(all_results)
        # print('all unique class values: ', list(set(unique_values)))

        print(f"\n $$$$$$$ evaluating in epoch:{epoch} $$$$$$$ \n")
        print('result: ',result_dict)
        # val_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"########## epoch:{epoch} mean_iou:{result_dict['mean_iou']} ############")
        # print(f"########## mean_iou using torchmetric library:{val_mean_iou} ############")
        
        return val_loss, result_dict['mean_iou']
