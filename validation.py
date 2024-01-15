import os
import time
import argparse
import datetime
import numpy as np
import sys

from utils.metric import hist_info, compute_score, cal_mean_iou
from utils.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from timm.utils import accuracy, AverageMeter

# next two are likely not needed
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_metric(results, config):
    hist = np.zeros((config.DATASET.num_classes, config.DATASET.num_classes))
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

def val_cityscape(epoch, val_loader, model, config):
    model.eval()
    sum_loss = 0
    all_results = []

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
            confusionMatrix, labeled, correct = hist_info(config.DATASET.num_classes, pred, gts)
            results_dict = {'hist': confusionMatrix, 'labeled': labeled, 'correct': correct}
            all_results.append(results_dict)
            
            sum_loss += loss
            del loss
            if idx % config.EVAL.val_print_stats == 0:
                print(f'sample {idx}')

        val_loss = sum_loss/len(val_loader)
        result_dict = compute_metric(all_results, config)

        print(f"\n ----------- evaluating in epoch:{epoch} ----------- \n")
        print('result: ',result_dict)
        print(f"#----------- epoch:{epoch} mean_iou:{result_dict['mean_iou']} -----------#")
        
        return val_loss, result_dict['mean_iou']


def val_ade(epoch, val_loader, model, config):
    model.eval()
    sum_loss = 0
    m_iou_batches = []
    all_results = []
    unique_values = []
    
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            imgs = sample['image']      #B, 3, 1024, 2048
            gts = sample['label']       #B, 1024, 2048
            
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

            loss, out = model(imgs, gts)

            # mean over multi-gpu result
            loss = torch.mean(loss)
            
            #miou using torchmetric library
            # m_iou = cal_mean_iou(out, gts)

            score = out[0]      #1, C, H, W --> C, H, W = 150, H, W
            score = torch.exp(score)    
            score = score.permute(1, 2, 0)  #H,W,C
            pred = score.argmax(2)  #H,W
            
            pred = pred.detach().cpu().numpy()
            gts = gts[0].detach().cpu().numpy() #1, H, W --> H, W
            confusionMatrix, labeled, correct = hist_info(config.DATASET.num_classes, pred, gts)
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
            if idx % config.EVAL.EVAL_PRINT_FREQ == 0:
                #pbar.set_description(print_str, refresh=True)
                print(f'sample {idx}')

        val_loss = sum_loss/len(val_loader)
        result_dict = compute_metric(all_results, config)
        # print('all unique class values: ', list(set(unique_values)))

        print(f"\n $$$$$$$ evaluating in epoch:{epoch} $$$$$$$ \n")
        print('result: ',result_dict)
        # val_mean_iou = np.mean(np.asarray(m_iou_batches))
        print(f"########## epoch:{epoch} mean_iou:{result_dict['mean_iou']} ############")
        # print(f"########## mean_iou using torchmetric library:{val_mean_iou} ############")
        
        return val_loss, result_dict['mean_iou']


def val_imagenet(epoch, data_loader, model, config):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        # images = images.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
        # target = target.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
        images = images.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)

        # compute output
        output = model(images)
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # need for distributed learning
        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.EVAL.EVAL_PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
    
@torch.no_grad()
def validation(epoch, val_loader, model, config):
    if config.DATASET.name == 'cityscapes':
        return val_cityscape(epoch, val_loader, model, config)
    elif config.DATASET.name == 'ade20k':
        return val_ade(epoch, val_loader, model, config)
    elif config.DATASET.name == 'imagenet':
        print('validation imagenet', type(val_loader))
        return val_imagenet(epoch, val_loader, model, config)
    else:
        raise NotImplementedError(f'Not yet supported {config.DATASET.name}')
