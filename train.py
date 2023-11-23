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



from models.builder import EncoderDecoder as segmodel

from dataloader.cityscapes_dataloader import CityscapesDataset
from validation import validation

from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR, PolyLR
from utils.metric import cal_mean_iou

from tensorboardX import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

def Main():
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')

    cudnn.benchmark = True
    seed = config.SYSTEM.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    cityscapes_train = CityscapesDataset(config, split='train')
    train_loader = DataLoader(cityscapes_train, batch_size=config.TRAIN.batch_size, shuffle=True, num_workers=config.TRAIN.num_workers, drop_last=True)
    print(f'total train: {len(cityscapes_train)} t_iteration:{len(train_loader)}')
    
    cityscapes_val = CityscapesDataset(config, split='val')
    val_loader = DataLoader(cityscapes_val, batch_size=1, shuffle=False, num_workers=config.TRAIN.num_workers)
    print(f'total val: {len(cityscapes_val)} v_iteration:{len(val_loader)}')
    

    save_log = os.path.join(config.WRITE.log_dir, str(run_id))
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    writer = SummaryWriter(save_log)


    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.IMAGE.background)
    model=segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.TRAIN.lr
    
    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, base_lr)
    
    if config.TRAIN.optimizer == 'AdamW': # Segformer original uses AdamW
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.TRAIN.weight_decay)
    elif config.TRAIN.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.TRAIN.momentum, weight_decay=config.TRAIN.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.TRAIN.nepochs * config.TRAIN.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.TRAIN.lr_power, total_iteration, config.TRAIN.niters_per_epoch * config.TRAIN.warm_up_epoch)
    print(f'lr_policy:{vars(lr_policy)}')
    
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
        sum_loss = 0
        m_iou_batches = []
        
        for idx, sample in enumerate(train_loader):
            imgs = sample['image']
            gts = sample['label']
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)  

            loss, out = model(imgs, gts)

            # mean over multi-gpu result
            loss = torch.mean(loss) 
            m_iou = cal_mean_iou(out, gts)
            m_iou_batches.append(m_iou)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Need to Change Lr Policy
            current_idx = (epoch- 1) * config.TRAIN.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

        
            sum_loss += loss
            print_str = 'Epoch {}/{}'.format(epoch, config.TRAIN.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.TRAIN.niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' miou=%.4e' %np.mean(np.asarray(m_iou_batches)) \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))+'\n'

            del loss
            if idx % config.TRAIN.train_print_stats == 0:
                print(f'{print_str}')

        train_loss = sum_loss/len(train_loader)
        train_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"########## epoch:{epoch} train_loss:{train_loss} t_miou:{train_mean_iou}############")
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_m_iou', train_mean_iou, epoch)

        #save model every 10 epochs before checkpoint_start_epoch
        if (epoch < config.MODEL.checkpoint_start_epoch) and (epoch % (config.MODEL.checkpoint_step*2) == 0):
            save_model(model, optimizer, epoch, run_id, config.WRITE.checkpoint_dir)
        #save model every 5 epochs after checkpoint_start_epoch
        elif (epoch >= config.MODEL.checkpoint_start_epoch) and (epoch % config.MODEL.checkpoint_step == 0) or (epoch == config.TRAIN.nepochs):
            save_model(model, optimizer, epoch, run_id, config.WRITE.checkpoint_dir)
        
        # compute val metrics
        val_loss, val_mean_iou = validation(epoch, val_loader, model, config)            
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_mIOU', val_mean_iou, epoch)
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
    config_filename = args.config.split('/')[-1].split('.')[0] 
    
    if config_filename == 'config_cityscapes':
        from configs.config_cityscapes import config
    else:
        raise NotImplementedError

    os.environ['MASTER_PORT'] = '169710'
    Main()
