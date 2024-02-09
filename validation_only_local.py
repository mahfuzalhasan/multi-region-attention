import scipy
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial

from configs.config_imagenet import config
from models.builder import EncoderDecoder as segmodel
from datasets import load_dataset ## for HF dataset



# def val_imagenet(data_loader, model, config):
#     criterion = torch.nn.CrossEntropyLoss()
#     model.eval()

#     batch_time = AverageMeter()
#     # loss_meter = AverageMeter()
#     # acc1_meter = AverageMeter()
#     # acc5_meter = AverageMeter()

#     top_1_total = 0
#     top_5_total = 0
#     total_loss = 0.0
#     total = 0

#     end = time.time()
#     for idx, (images, target) in enumerate(data_loader):
#         images = images.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
#         target = target.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
#         # images = images.to('cuda', non_blocking=True)
#         # target = target.to('cuda', non_blocking=True)

#         # compute output
#         output = model(images)
#         # measure accuracy and record loss
#         loss = criterion(output, target)
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         top_1_total += acc1.item()
#         top_5_total += acc5.item()
#         total_loss += loss.item() * images.size(0)
#         total += images.size(0)

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.EVAL.EVAL_PRINT_FREQ == 0:
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

#             # batch_total_loss_tensor = torch.tensor([total_loss], device=dist.get_rank())
#             # batch_total_tensor = torch.tensor([total], device=dist.get_rank())
#             # batch_top1_tensor = torch.tensor([top_1_total], device=dist.get_rank())
#             # batch_top5_tensor = torch.tensor([top_5_total], device=dist.get_rank())

#             # # # Aggregate counts across all GPUs
#             # # dist.all_reduce(batch_total_loss_tensor, op=dist.ReduceOp.SUM)
#             # # dist.all_reduce(batch_total_tensor, op=dist.ReduceOp.SUM)
#             # # dist.all_reduce(batch_top1_tensor, op=dist.ReduceOp.SUM)
#             # # dist.all_reduce(batch_top5_tensor, op=dist.ReduceOp.SUM)

#             # batch_avg_loss = batch_total_loss_tensor.item() / batch_total_tensor.item()
#             # batch_top1_accuracy = batch_top1_tensor.item() / batch_total_tensor.item() * 100
#             # batch_top5_accuracy = batch_top5_tensor.item() / batch_total_tensor.item() * 100

#             # if dist.get_rank()==0:
#             print(
#                 f'Test: [{idx}/{len(data_loader)}]\t'
#                 f'Batch Time Avg ({batch_time.avg:.3f})\t'
#                 f'Loss avg  ({batch_avg_loss:.4f})\t'
#                 f'Acc@1 avg ({batch_top1_accuracy:.3f})\t'
#                 f'Acc@5 avg ({batch_top5_accuracy:.3f})\t'
#                 f'Mem {memory_used:.0f}MB')
#     print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
#     return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
#     ###### validation done ########  
#     # total_loss_tensor = torch.tensor([total_loss], device=dist.get_rank())
#     # total_tensor = torch.tensor([total], device=dist.get_rank())
#     # top1_tensor = torch.tensor([top_1_total], device=dist.get_rank())
#     # top5_tensor = torch.tensor([top_5_total], device=dist.get_rank())

#     # # Aggregate counts across all GPUs
#     # dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
#     # dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
#     # dist.all_reduce(top1_tensor, op=dist.ReduceOp.SUM)
#     # dist.all_reduce(top5_tensor, op=dist.ReduceOp.SUM)

#     # if dist.get_rank() == 0:  # Optionally, only on the master node
#     # avg_loss = total_loss_tensor.item() / total_tensor.item()
#     # top1_accuracy = top1_tensor.item() / total_tensor.item() * 100
#     # top5_accuracy = top5_tensor.item() / total_tensor.item() * 100
#     # print(f'Validation Loss: {avg_loss:.4f}, Top-1 Accuracy: {top1_accuracy:.2f}%, Top-5 Accuracy: {top5_accuracy:.2f}%')
#     return top1_accuracy, top5_accuracy, avg_loss
#     # else:
#     #     return None, None, None
    
def val_imagenet(data_loader, model, config):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    top_1_total = 0
    top_5_total = 0
    total_loss = 0.0
    total = 1

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
        
        print(f'Batch: {idx}, acc1/iter: {acc1}, acc5/iter: {acc5}')

        # need for distributed learning
        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        top_1_total += acc1.item()
        top_5_total += acc5.item()
        total_loss += loss.item() * images.size(0)
        
        print(f'top_1_total: {top_1_total}, top_5_total: {top_5_total}, total: {total}')
        
        # total += images.size(0)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.EVAL.EVAL_PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            top_1_batch = (top_1_total / total) # batches seen so far
            top_5_batch = (top_5_total / total) 

            print(f'top_1_batch: {top_1_batch}, top_5_batch: {top_5_batch}')

            print(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
        
        total += 1
    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    
    final_top_1 = (top_1_total / total)
    final_top_5 = (top_5_total / total)

    print(f'final_top_1: {final_top_1}, final_top_5: {final_top_5}')
    print(f'top_1_total: {top_1_total} top_5_total: {top_5_total} total_loss: {total_loss} total: {total}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

####################### WHAT TRANSFORM FOR IMAGENET VALIDATION ?????????????
# def build_transform(is_train, config):
#     resize_im = config.DATASET.IMG_SIZE > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=config.DATASET.IMG_SIZE,
#             is_training=True,
#             color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
#             auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATASET.INTERPOLATION,
#         )
#         if dist.get_rank()==0:
#             print("transform: ",transform)
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(config.DATASET.IMG_SIZE, padding=4)
#         return transform

def build_transform(is_train, config):
    resize_im = config.DATASET.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATASET.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATASET.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATASET.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATASET.IMG_SIZE)
            t.append(
                transforms.Resize(size)
                # transforms.Resize(size, interpolation=_pil_interp(config.DATASET.INTERPOLATION)),
            )
            t.append(transforms.CenterCrop(config.DATASET.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATASET.IMG_SIZE, config.DATASET.IMG_SIZE))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_dataset(is_train, config):
    
    transform = build_transform(is_train, config) ### not used for validation
    if config.DATASET.NAME == 'imagenet':
        val_data_dir = './data/imagenet/validation/'      
        
        dataset = datasets.ImageFolder(val_data_dir, transform=transform)  
        nb_classes = 1000
        # dataset = load_dataset('imagenet-1k')
        
        # if is_train:
        #     hf_dataset = hf_dataset['train']
        # else:
        #     hf_dataset = hf_dataset['validation']
    else:
        raise NotImplementedError

    
    return dataset, nb_classes


def build_loader(config):
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f'dataset_val: {dataset_val}')

    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()

    # if config.DATASET.ZIP_MODE and config.DATASET.CACHE_MODE == 'part':
    #     indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
    #     sampler_train = SubsetRandomSampler(indices)
    # else:
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
    #     )

    # indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # # indices = np.arange(global_rank, len(dataset_val), num_tasks)
    # sampler_val = SubsetRandomSampler(indices)

    # sampler_val = torch.utils.data.DistributedSampler(
    #     dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    # )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=config.DATASET.BATCH_SIZE,
    #     num_workers=config.DATASET.NUM_WORKERS,
    #     pin_memory=config.DATASET.PIN_MEMORY,
    #     drop_last=True,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=False
    )
    print("Val dataloading completed")
    
    ################ Mixup / Cutmix NOT USED for validation????? CHECK THIS LATER ####
    # # setup mixup / cutmix
    # mixup_fn = None
    # mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
    #         prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
    #         label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.DATASET.NUM_CLASSES)

    return dataset_val, data_loader_val


@torch.no_grad()
def validation(val_loader, model, config):
    # if config.DATASET.NAME == 'cityscapes':
    #     return val_cityscape(epoch, val_loader, model, config)
    # elif config.DATASET.NAME == 'ade20k':
    #     return val_ade(epoch, val_loader, model, config)
    if config.DATASET.NAME == 'imagenet' or config.DATASET.NAME == 'tiny-imagenet':
        print('validation imagenet', type(val_loader))
        return val_imagenet(val_loader, model, config)
    else:
        raise NotImplementedError(f'Not yet supported {config.DATASET.name}')
    

def build_model(config):

    # config network and criterion
    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    criterion = torch.nn.CrossEntropyLoss()
    model = segmodel(cfg=config, criterion=criterion, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

if __name__ == '__main__':
    # data_dir = '../data/imagenet/validation/'

    # dataset = datasets.ImageFolder(data_dir)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    # print(f'dataset: {dataset}')
    # print(f'dataloader length: {len(dataloader)}')

    model = build_model(config)
    # print(config)
    model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=config.SYSTEM.device_ids)


    # TODO: Load checkpoint
    saved_checkpoint_path = '/home/ma906813/project2023/multi-region-attention/results/model_best_02-02-24_2146.pth'
    state_dict = torch.load(saved_checkpoint_path)
    model.load_state_dict(state_dict['model'])
    # optimizer.load_state_dict(state_dict['optimizer'])
    # lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
    # starting_epoch = state_dict['epoch']
    # max_accuracy = state_dict['max_accuracy']
    run_id = state_dict['run_id']

    # TODO: build dataloader
    dataset_val, data_loader_val = build_loader(config)
    print(f'dataset_val length: {len(dataset_val)}')
    print(f'data_loader_val length: {len(data_loader_val)}')

    # TODO: validation
    with torch.no_grad():
        acc1, acc5, v_loss = validation(data_loader_val, model, config)