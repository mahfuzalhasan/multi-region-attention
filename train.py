import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from models.builder import EncoderDecoder as segmodel
from dataloader.imagenet.build import build_loader
# from validation import validation
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR, PolyLR
# from utils.metric import cal_mean_iou
from utils.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

from tensorboardX import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

def Main(args):
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')

    config_filename = args.config.split('/')[-1].split('.')[0] 
    print('cnfig_filename: ',config_filename)    
    if config_filename == 'imagenet':
        from configs.config_imagenet import config
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    else:
        raise NotImplementedError

    save_log = os.path.join(config.WRITE.log_dir, str(run_id))
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    writer = SummaryWriter(save_log)

    cudnn.benchmark = True
    seed = config.SYSTEM.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model=segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # group weight and config optimizer
    base_lr = config.TRAIN.BASE_LR

    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    
    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, base_lr)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    # if config.TRAIN.optimizer == 'AdamW': # Segformer original uses AdamW
    #     optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.TRAIN.weight_decay)
    # elif config.TRAIN.optimizer == 'SGDM':
    #     optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.TRAIN.momentum, weight_decay=config.TRAIN.weight_decay)
    # else:
    #     raise NotImplementedError

    # config lr policy
    
    print("multigpu training")
    print('device ids: ',config.SYSTEM.device_ids)
    model = nn.DataParallel(model, device_ids = config.SYSTEM.device_ids)
    model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

    print(f'############ begin training ################# \n')

    starting_epoch = 1
    if config.TRAIN.resume_train:
        print('Loading model to resume train')
        state_dict = torch.load(config.TRAIN.resume_model_path)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        starting_epoch = state_dict['epoch']
        print('resuming training with model: ', config.TRAIN.resume_model_path)
    
    for epoch in range(starting_epoch, config.TRAIN.nepochs):
        model.train()
        optimizer.zero_grad()
        num_steps = len(data_loader_train)
        sum_loss = 0
        m_iou_batches = []

        n_params = count_parameters(model)
        print(f'params: {n_params} steps: {num_steps}')
        

        # exit()
        
        for idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            targets = targets.to(f'cuda:{model.device_ids[0]}', non_blocking=True)  

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            # loss, out = model(imgs, gts)

            

            outputs = model(samples)
            # print(f"samples:{samples.shape} outputs:{outputs.shape} loss_part:{loss_part}")
            loss = criterion(outputs, targets)
            # print(f"loss: {loss}")
            # mean over multi-gpu result
            # loss = torch.mean(loss) 
            optimizer.zero_grad()
            loss.backward()
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())

            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

            lr = optimizer.param_groups[0]["lr"]
            sum_loss += loss.item()
            print_str = 'Epoch {}/{}'.format(epoch, config.TRAIN.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, len(data_loader_train)) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.TRAIN.train_print_stats == 0:
                print(f'{print_str}')

        train_loss = sum_loss/len(train_loader)
        train_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"########## epoch:{epoch} train_loss:{train_loss}############")
        writer.add_scalar('train_loss', train_loss, epoch)

        #save model every 10 epochs before checkpoint_start_epoch
        if (epoch < config.MODEL.checkpoint_start_epoch) and (epoch % (config.MODEL.checkpoint_step*2) == 0):
            save_model(model, optimizer, epoch, run_id, config.WRITE.checkpoint_dir)
        #save model every 5 epochs after checkpoint_start_epoch
        elif (epoch >= config.MODEL.checkpoint_start_epoch) and (epoch % config.MODEL.checkpoint_step == 0) or (epoch == config.TRAIN.nepochs):
            save_model(model, optimizer, epoch, run_id, config.WRITE.checkpoint_dir)
        
        # compute val metrics
        # val_loss, val_mean_iou = validation(epoch, val_loader, model, config)            
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('val_mIOU', val_mean_iou, epoch)
        # print(f't_loss:{train_loss:.4f} v_loss:{val_loss:.4f} val_mIOU:{val_mean_iou:.4f}')

def save_model(model, optimizer, epoch, run_id, checkpoint_dir):
    save_dir = os.path.join(checkpoint_dir, str(run_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
    states = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
    }
    torch.save(states, save_file_path)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Segformer Training')
    parser.add_argument('config', help='dataset specific train config file path, more details can be found in configs/')

    args = parser.parse_args()

    os.environ['MASTER_PORT'] = '169710'
    Main(args)
