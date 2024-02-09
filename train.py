import os
import time
import argparse
import datetime
import numpy as np
import sys
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist



import torch
from  torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from models.builder import EncoderDecoder as segmodel
from dataloader.imagenet.build import build_loader
from validation import validation
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
    
    
    config_filename = args.config.split('/')[-1].split('.')[0] 
    print('config_filename: ',config_filename)    
    if config_filename == 'imagenet':
        from configs.config_imagenet import config
        print(f'config loaded')
    else:
        raise NotImplementedError

    config.SYSTEM.DEVICE_IDS = [i for i in range(args.devices)]
    config.DATASET.NAME = args.dataset
    config.DATASET.BATCH_SIZE = args.batchsize

    dist_backend = 'nccl'
    torch.distributed.init_process_group(backend=dist_backend)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.distributed.barrier()     ## this keeps checkpoints during training for all processes to catch up and sync
    
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.cuda.synchronize()
    run_id = datetime.datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ new_run_id:{run_id} $$$$$$$$$$$$$')

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # config network and criterion
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model=segmodel(cfg=config, criterion=criterion, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(dist.get_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    total_batch_size = config.DATASET.BATCH_SIZE * dist.get_world_size()
    config.TRAIN.BASE_LR = config.TRAIN.BASE_LR * total_batch_size / 510.0
    config.TRAIN.WARMUP_LR = config.TRAIN.WARMUP_LR * total_batch_size / 510.0
    config.TRAIN.MIN_LR = config.TRAIN.MIN_LR * total_batch_size / 510.0

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # group weight and config optimizer
    base_lr = config.TRAIN.BASE_LR

    print(f'rank:{dist.get_rank()} base_lr:{config.TRAIN.BASE_LR} w_lr:{config.TRAIN.WARMUP_LR} min_lr:{config.TRAIN.MIN_LR}')

    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    
    # params_list = []
    # params_list = group_weight(params_list, model, nn.BatchNorm2d, base_lr)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if dist.get_rank()==0:
        print(f'############ begin training ################# \n')

    starting_epoch = 0
    max_accuracy = None
    if config.TRAIN.RESUME_TRAIN:
        print('Loading model to resume train')
        state_dict = torch.load(config.TRAIN.RESUME_MODEL_PATH)
        model.module.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        starting_epoch = state_dict['epoch'] + 1        # Start from next epoch
        max_accuracy = state_dict['max_accuracy']
        run_id = state_dict['run_id']
        print('resuming training with model: ', config.TRAIN.RESUME_MODEL_PATH)
        print('old_run_id: ', run_id)

    if dist.get_rank()==0:
        print(f"\n #training:{len(dataset_train)} #val:{len(dataset_val)}")
        print(f"\n iteration in one epoch: len_dl::: train:{len(data_loader_train)} val:{len(data_loader_val)}")
        print(f"\n batch size:{config.DATASET.BATCH_SIZE} \n")
    
        save_log = os.path.join(config.WRITE.LOG_DIR, str(run_id))
        os.makedirs(save_log, exist_ok=True)
        writer = SummaryWriter(save_log)

    n_params = count_parameters(model)
    if dist.get_rank()==0:
        print(f'#params of the model: {n_params}')
    
    torch.cuda.synchronize()
    start_time = time.time()
    for epoch in range(starting_epoch, config.TRAIN.EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        num_steps = len(data_loader_train)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        epoch_loss = 0.0
        

        torch.cuda.synchronize()
        start = time.time()
        end = time.time()
        
        for idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(dist.get_rank(), non_blocking=True)
            targets = targets.to(dist.get_rank(), non_blocking=True) 
           
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss.backward()

            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

            epoch_loss += loss.item()
            loss_meter.update(loss.item(), targets.size(0))
            batch_time.update(time.time() - end)
            norm_meter.update(grad_norm)

            if idx % config.TRAIN.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                if dist.get_rank() == 0:
                    print(
                        f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'batch time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            end = time.time()
           
            # if idx%5==0:
            #     break
        ############ epoch ends here###############
        dist.barrier()
        t_loss = loss_meter.avg
        epoch_time = time.time() - start
        if dist.get_rank()==0:
            print(f'Training loss epoch:{epoch}::: {t_loss}')
            print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        if max_accuracy is None:
            max_accuracy = 0.0
        
        start_val = time.time()
        with torch.no_grad():
            print(f'rank:{dist.get_rank()} entering validation')
            acc1, acc5, v_loss = validation(epoch, data_loader_val, model, config)
            print(f'rank:{dist.get_rank()} validation acc:{acc1}')
        
        # Save model with best accuracy as model_best
        if dist.get_rank()==0:
            model_save_start = time.time()
            if acc1 > max_accuracy:
                max_accuracy = max(max_accuracy, acc1)
                save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, config.WRITE.CHECKPOINT_DIR, best=True)
            #save model every 25 epochs before checkpoint_start_epoch = 200
            if (epoch <= config.MODEL.CHECKPOINT_START_EPOCH) and (epoch % (config.MODEL.CHECKPOINT_STEP) == 0) and (epoch>0):
                save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, config.WRITE.CHECKPOINT_DIR)
            #save model every 10 epochs after checkpoint_start_epoch=200. Save last one too. 
            elif (epoch > config.MODEL.CHECKPOINT_START_EPOCH) and (epoch % config.MODEL.CHECKPOINT_STEP_LATER == 0) or (epoch == config.TRAIN.EPOCHS-1):
                save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, config.WRITE.CHECKPOINT_DIR)
            model_save_time = time.time() - model_save_start
            print(f"EPOCH {epoch} model save takes {datetime.timedelta(seconds=int(model_save_time))}")

            ### Writing to Tensorboard
            writer_start = time.time()
            writer.add_scalar('val_max_acc', max_accuracy, epoch)   
            writer.add_scalar('val_loss', v_loss, epoch)
            writer.add_scalar('val_acc_1', acc1, epoch)
            writer.add_scalar('val_acc_5', acc5, epoch)
            # writer.add_scalar('train_max_acc', training_max_acc, epoch)
            writer.add_scalar('train_loss', t_loss, epoch)
            # writer.add_scalar('train_acc_1', t_acc1, epoch)
            # writer.add_scalar('train_acc_5', t_acc5, epoch)
            writer_time = time.time() - writer_start
            print(f"EPOCH {epoch} writer takes {datetime.timedelta(seconds=int(writer_time))}")

            val_epoch_time = time.time() - start_val
            print(f"EPOCH {epoch} val takes {datetime.timedelta(seconds=int(val_epoch_time))}")
            print(f'\n ###### stats after epoch :{epoch} ######### \n')
            print(f't_loss:{t_loss:.4f} v_loss:{v_loss:.4f}') 
            print(f'v_acc1:{acc1:.4f} v_acc5:{acc5:.4f}')
            print(f'\n ######## epoch {epoch} is completed ########### \n')
    ##### Training Ends Here##############
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'rank:{dist.get_rank()} - Total Training time {total_time_str}')
    dist.destroy_process_group()

def save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, CHECKPOINT_DIR, best=False):
    save_dir = os.path.join(CHECKPOINT_DIR, str(run_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_file_path = os.path.join(save_dir, 'model_{}.pth'.format(epoch))
    if best:
        save_file_path = os.path.join(save_dir, 'model_best.pth')

    save_state = {'model': model.module.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'run_id':str(run_id)}
    torch.save(save_state, save_file_path)
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Segformer Training')
    parser.add_argument('config', help='dataset specific train config file path, more details can be found in configs/')

    parser.add_argument('--devices', default=1, type=int, help='gpu devices')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size for single gpu')
    args = parser.parse_args()

    print("code running")
    
    Main(args)


    

