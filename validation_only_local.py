import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import argparse

from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from functools import partial
from utils.utils import reduce_tensor

from configs.config_imagenet import config
from models.builder import EncoderDecoder as segmodel
from datasets import load_dataset ## for HF dataset

from dataloader.imagenet.HFDataset import HFDataset


    
def val_imagenet(data_loader, model, config):
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
    transform = build_transform(is_train, config)
    if config.DATASET.NAME == 'imagenet':
        hf_dataset = load_dataset('imagenet-1k')
        nb_classes = 1000
        if is_train:
            hf_dataset = hf_dataset['train']
        else:
            hf_dataset = hf_dataset['validation']
    elif config.DATASET.NAME == 'tiny-imagenet':
        hf_dataset = load_dataset('Maysee/tiny-imagenet')
        nb_classes = 200
        if is_train:
            hf_dataset = hf_dataset['train']
        else:
            hf_dataset = hf_dataset['valid']
    else:
        raise NotImplementedError
    # Wrap Hugging Face dataset with PyTorch Dataset to apply transformations
    dataset = HFDataset(hf_dataset, transform=transform)
    return dataset, nb_classes


def build_loader(config):
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f'length dataset_val: {len(dataset_val)}')

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
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
    criterion = torch.nn.CrossEntropyLoss()
    model = segmodel(cfg=config, criterion=criterion, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Segformer Training')
    parser.add_argument('config', help='dataset specific train config file path, more details can be found in configs/')

    parser.add_argument('--devices', default=1, type=int, help='gpu devices')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size for single gpu')
    args = parser.parse_args()


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


    model = build_model(config)
    model.to(dist.get_rank())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])


    saved_checkpoint_path = '/project/results/saved_models/02-14-24_2014/model_best.pth'
    # saved_checkpoint_path = '/home/ma906813/project2023/multi-region-attention/model_best.pth'
    state_dict = torch.load(saved_checkpoint_path)
    model.module.load_state_dict(state_dict['model'])
    run_id = state_dict['run_id']
    epoch = state_dict['epoch']
    max_accuarcy = state_dict['max_accuracy']
    print(f'best model from epoch:{epoch} max acc:{max_accuarcy}')
    exit()
    
    dataset_val, data_loader_val = build_loader(config)
    if dist.get_rank()==0:
        print(f'best model loaded from epoch:{epoch}')
        print(f'dataset_val length: {len(dataset_val)}')
        print(f'data_loader_val length: {len(data_loader_val)}')

    with torch.no_grad():
        acc1, acc5, v_loss = validation(data_loader_val, model, config)