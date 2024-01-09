import os
import time
import argparse
import datetime
import numpy as np
import sys


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
    run_id = datetime.datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')

    config_filename = args.config.split('/')[-1].split('.')[0] 
    print('cnfig_filename: ',config_filename)    
    if config_filename == 'imagenet':
        from configs.config_imagenet import config
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    else:
        raise NotImplementedError
    
    print(f"\n #training:{len(dataset_train)} #val:{len(dataset_val)}")
    print(f"\n it in one epoch: len_dl::: train:{len(data_loader_train)} val:{len(data_loader_val)}")
    print(f"\n batch size:{config.DATASET.BATCH_SIZE} \n")
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
    
    print("multigpu training")
    print('device ids: ',config.SYSTEM.device_ids)
    model = nn.DataParallel(model, device_ids = config.SYSTEM.device_ids)
    model.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

    print(f'############ begin training ################# \n')

    starting_epoch = 0
    max_accuracy = None
    if config.TRAIN.resume_train:
        print('Loading model to resume train')
        state_dict = torch.load(config.TRAIN.resume_model_path)
        model.module.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        starting_epoch = state_dict['epoch']
        max_accuracy = state_dict['max_accuracy']
        old_run_id = state_dict['run_id']
        print('resuming training with model: ', config.TRAIN.resume_model_path)
        print('resuming experiment from: ', old_run_id)

    n_params = count_parameters(model)
    print(f'#params of the model: {n_params}')

    start_time = time.time()
    for epoch in range(starting_epoch, config.TRAIN.nepochs):
        model.train()
        optimizer.zero_grad()
        
        num_steps = len(data_loader_train)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()

        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        
        start = time.time()
        end = time.time()
        
        for idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            targets = targets.to(f'cuda:{model.device_ids[0]}', non_blocking=True) 
            # print(f'target initial: {targets.size()}') 
            initial_targets = targets.clone()
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

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            norm_meter.update(grad_norm)
            # print(f'output:::::{outputs.size()} target:{targets.size()}')
            t_acc1, t_acc5 = accuracy(outputs, initial_targets, topk=(1, 5))
            acc1_meter.update(t_acc1.item(), initial_targets.size(0))
            acc5_meter.update(t_acc5.item(), initial_targets.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % config.TRAIN.train_print_stats == 0:
                lr = optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                print(
                    f'Train: [{epoch}/{config.TRAIN.nepochs}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'batch time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                    f'mem {memory_used:.0f}MB')
            # if idx%5==0:
            #     break
        t_loss = loss_meter.avg
        t_acc1 = acc1_meter.avg
        t_acc5 = acc5_meter.avg
        print(f' Training::::: Acc@1 {t_acc1:.3f} Acc@5 {t_acc5:.3f}')
        training_max_acc = 0.0
        training_max_acc = max(training_max_acc, t_acc1)
        writer.add_scalar('train_max_acc', training_max_acc, epoch)
        writer.add_scalar('train_loss', t_loss, epoch)
        writer.add_scalar('train_acc_1', t_acc1, epoch)
        writer.add_scalar('train_acc_5', t_acc5, epoch)

        epoch_time = time.time() - start
        print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        if max_accuracy is None:
            max_accuracy = 0.0
        #save model every 50 epochs before checkpoint_start_epoch = 200
        if (epoch <= config.MODEL.checkpoint_start_epoch) and (epoch % (config.MODEL.checkpoint_step*2) == 0) and (epoch>0):
            save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, config.WRITE.checkpoint_dir)
        #save model every 25 epochs after checkpoint_start_epoch=200. Save last one too. 
        elif (epoch > config.MODEL.checkpoint_start_epoch) and (epoch % config.MODEL.checkpoint_step == 0) or (epoch == config.TRAIN.nepochs-1):
            save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, config.WRITE.checkpoint_dir)
        
        start_val = time.time()
        with torch.no_grad():
            acc1, acc5, v_loss = validation(epoch, data_loader_val, model, config)
        
        # Save model with best accuracy as model_best
        if acc1 > max_accuracy:
            max_accuracy = max(max_accuracy, acc1)
            save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, config.WRITE.checkpoint_dir, best=True)

        writer.add_scalar('val_max_acc', max_accuracy, epoch)   
        writer.add_scalar('val_loss', v_loss, epoch)
        writer.add_scalar('val_acc_1', acc1, epoch)
        writer.add_scalar('val_acc_5', acc5, epoch)
        val_epoch_time = time.time() - start_val
        print(f"EPOCH {epoch} val takes {datetime.timedelta(seconds=int(val_epoch_time))}")
        print(f'\n ###### stats after epoch :{epoch} ######### \n')
        print(f't_loss:{t_loss:.4f} v_loss:{v_loss:.4f}') 
        print(f't_acc1:{t_acc1:.4f} t_acc5: {t_acc5:.4f} v_acc1:{acc1:.4f} v_acc5:{acc5:.4f}')
        print(f'\n ######## epoch {epoch} is completed ########### \n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total Training time {total_time_str}')

def save_model(model, optimizer, lr_scheduler, epoch, run_id, max_accuracy, checkpoint_dir, best=False):
    save_dir = os.path.join(checkpoint_dir, str(run_id))
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

    args = parser.parse_args()

    os.environ['MASTER_PORT'] = '169710'
    Main(args)
