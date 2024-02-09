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
import torch
import torch.distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_metric(results, config):
    hist = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
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

    top_1_total = 0
    top_5_total = 0
    total_loss = 0.0
    total = 0

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

        #### calculation with avg_meter
        total_loss += loss.item() * images.size(0)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        total += 1
        top_1_total += acc1.item()
        top_5_total += acc5.item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.EVAL.EVAL_PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            # top_1_batch = (top_1_total / total)
            # top_5_batch = (top_5_total / total)
            
            # batch_total_loss_tensor = torch.tensor([total_loss], device=dist.get_rank())
            batch_total_tensor = torch.tensor([total], device=dist.get_rank())
            csum_batch_top1_tensor = torch.tensor([top_1_total], device=dist.get_rank())
            csum_batch_top5_tensor = torch.tensor([top_5_total], device=dist.get_rank())

            # Aggregate counts across all GPUs
            dist.all_reduce(batch_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(csum_batch_top1_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(csum_batch_top5_tensor, op=dist.ReduceOp.SUM)


            ravg_batch_top1_accuracy = csum_batch_top1_tensor.item() / batch_total_tensor.item()
            ravg_batch_top5_accuracy = csum_batch_top5_tensor.item() / batch_total_tensor.item()

            if dist.get_rank()==0:
                print(                                                  ## print from manual calc
                    f'#From Manual Calc:: Test: [{idx}/{len(data_loader)}]\t'
                    f'Batch Time Avg ({batch_time.avg:.3f})\t'
                    f'Acc@1 avg ({ravg_batch_top1_accuracy:.3f})\t'
                    f'Acc@5 avg ({ravg_batch_top5_accuracy:.3f})\t'
                    f'Mem {memory_used:.0f}MB')
                print(                                                  ## print from AverageMeter
                    f'#From AverageMeter:: Test: [{idx}/{len(data_loader)}]\t'
                    f'Loss avg  ({loss_meter.avg:.4f})\t'
                    f'Acc@1 avg ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 avg ({acc5_meter.avg:.3f})\t')

    # batch_total_loss_tensor = torch.tensor([total_loss], device=dist.get_rank())
    batch_total_tensor = torch.tensor([total], device=dist.get_rank())
    csum_batch_top1_tensor = torch.tensor([top_1_total], device=dist.get_rank())
    csum_batch_top5_tensor = torch.tensor([top_5_total], device=dist.get_rank())

    # Aggregate counts across all GPUs
    dist.all_reduce(batch_total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(csum_batch_top1_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(csum_batch_top5_tensor, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:  # Optionally, only on the master node
        top1_accuracy = csum_batch_top1_tensor.item() / batch_total_tensor.item()
        top5_accuracy = csum_batch_top5_tensor.item() / batch_total_tensor.item()

        print(f'Validation Loss: {loss_meter.avg:.4f}, Top-1 Accuracy: {top1_accuracy:.2f}%, Top-5 Accuracy: {top5_accuracy:.2f}%')
        print(f'accuracy from avg meter::: acc1:{acc1_meter.avg:.3f} acc5:{acc5_meter.avg:.3f}')
        return top1_accuracy, top5_accuracy, loss_meter.avg
    else:
        return None, None, None
    
@torch.no_grad()
def validation(epoch, val_loader, model, config):
    if config.DATASET.NAME == 'cityscapes':
        return val_cityscape(epoch, val_loader, model, config)
    elif config.DATASET.NAME == 'ade20k':
        return val_ade(epoch, val_loader, model, config)
    elif config.DATASET.NAME == 'imagenet' or config.DATASET.NAME == 'tiny-imagenet':
        print('validation imagenet', type(val_loader))
        return val_imagenet(val_loader, model, config)
    else:
        raise NotImplementedError(f'Not yet supported {config.DATASET.name}')
