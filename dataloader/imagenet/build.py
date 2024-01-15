# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
# from timm.data.transforms import _pil_interp

from .samplers import SubsetRandomSampler

from datasets import load_dataset
from .HFDataset import HFDataset


def build_loader(config):
    # config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    # config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    print(f'num task:{num_tasks} global_rank:{global_rank}')

    # num_tasks = 4
    # global_rank = 0
    if config.DATASET.ZIP_MODE and config.DATASET.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    # indices = np.arange(global_rank, len(dataset_val), num_tasks)
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATASET.BATCH_SIZE,
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATASET.NUM_WORKERS,
        pin_memory=config.DATASET.PIN_MEMORY,
        drop_last=False
    )
    print("dataloader completed")

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.DATASET.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


# def build_dataset(is_train, config):
#     transform = build_transform(is_train, config)
#     print("is_train: ",is_train)
#     prefix = 'train' if is_train else 'val'
#     root = os.path.join(config.DATASET.root, prefix)
#     print("data root: ",root)

#     dataset = datasets.ImageFolder(root, transform=transform)
#     print('loader ', dataset.loader)
#     print("completed -- ---- --- -- -- ")
#     nb_classes = 1000
#     print('type of dataset: ',type(dataset))
#     return dataset, nb_classes

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)

    hf_dataset = load_dataset('Maysee/tiny-imagenet')
    # hf_dataset = load_dataset('imagenet-1k')
    if is_train:
        hf_dataset = hf_dataset['train']
    else:
        hf_dataset = hf_dataset['valid']
    # Wrap Hugging Face dataset with PyTorch Dataset to apply transformations
    dataset = HFDataset(hf_dataset, transform=transform)
    nb_classes = 200  # Number of classes for ImageNet, 200 for tiny
    return dataset, nb_classes



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
        # print("transform: ",transform)
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
    # print('imagenet mean and std: ',IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    return transforms.Compose(t)
