import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C

""" General """


"""System Config"""   
C.SYSTEM = edict()
C.SYSTEM.seed = 12345
remoteip = os.popen('pwd').read()
C.SYSTEM.root_dir = os.getcwd()
C.SYSTEM.abs_dir = osp.realpath(".")
#C.SYSTEM.device_ids = [0] # for mahdi (lab-pc)
C.SYSTEM.device_ids = [0, 1] # for mahdi (newton)
# C.SYSTEM.device_ids = [0, 1, 2, 3] # for FICS
C.SYSTEM.device_ids = [0, 1, 2, 3, 4, 5, 6, 7] # for nautilus


"""Dataset Config"""
C.DATASET = edict()
C.DATASET.name = 'imagenet'
C.DATASET.root = '/home/abjawad/Documents/GitHub/multi-region-attention/data/imagenet/ILSVRC/Data/CLS-LOC'
# C.DATASET.DATA_PATH = osp.join(C.SYSTEM.root_dir, 'data/Cityscapes')
C.DATASET.mode = 'RGB'
# Input image size
C.DATASET.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
C.DATASET.INTERPOLATION = 'bicubic'
# C.DATASET.dataset_config_path = osp.join(C.SYSTEM.root_dir, 'dataloader/cityscapes_rgbd_config.yaml')

C.DATASET.NUM_CLASSES = 1000  #for imagenet

# Cache Data in Memory, could be overwritten by command line argument
C.DATASET.CACHE_MODE = 'part'

####################
# Batch size for a single GPU, could be overwritten by command line argument
C.DATASET.BATCH_SIZE = 100
# Path to dataset, could be overwritten by command line argument
C.DATASET.DATA_PATH = ''
# Dataset name
C.DATASET.DATASET = 'imagenet'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
C.DATASET.ZIP_MODE = False

# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
C.DATASET.PIN_MEMORY = True
# Number of data loading threads
C.DATASET.NUM_WORKERS = 8
##################



"""Image Config"""
C.IMAGE = edict()
C.IMAGE.image_height = 224
C.IMAGE.image_width = 224


""" Augmentation """
# Augmentation settings
# -----------------------------------------------------------------------------
C.AUG = edict()
# Color jitter factor
C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
C.AUG.REPROB = 0.25
# Random erase mode
C.AUG.REMODE = 'pixel'
# Random erase count
C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
C.AUG.MIXUP_MODE = 'batch'



""" Model Config"""
C.MODEL = edict()
C.MODEL.backbone = 'mra_tiny'
C.MODEL.pretrained_model = None #osp.join(C.SYSTEM.root_dir, 'pretrained/mit_b2_imagenet.pth')
C.MODEL.heads = [3, 6, 12, 24]
C.MODEL.decoder = 'ClassificationHead'#'MLPDecoder'
C.MODEL.decoder_embed_dim = 768
C.MODEL.checkpoint_start_epoch = 3
C.MODEL.checkpoint_step = 1
C.MODEL.NAME = 'mra_tiny'
C.MODEL.LABEL_SMOOTHING = 0.1
# Dropout rate
C.MODEL.DROP_RATE = 0.0
# Drop path rate
C.MODEL.DROP_PATH_RATE = 0.1
C.MODEL.GSA = False


"""Train Config"""
C.TRAIN = edict()
C.TRAIN.BASE_LR = 5e-4
C.TRAIN.WARMUP_LR = 5e-7
C.TRAIN.MIN_LR = 5e-6

C.TRAIN.ACCUMULATION_STEPS = 0

C.TRAIN.nepochs = 300
C.TRAIN.num_workers = 4

C.TRAIN.warm_up_epoch = 20
C.TRAIN.CLIP_GRAD = 5.0
C.TRAIN.WEIGHT_DECAY = 0.05

C.TRAIN.fix_bias = True
C.TRAIN.bn_eps = 1e-3
C.TRAIN.bn_momentum = 0.1
C.TRAIN.train_print_stats = 400
C.TRAIN.resume_train = False 
C.TRAIN.resume_model_path = osp.join(C.SYSTEM.root_dir, 'results/saved_models/12-29-23_2141/model_195.pth')

# LR scheduler
C.TRAIN.LR_SCHEDULER = edict()
C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
C.TRAIN.OPTIMIZER = edict()
C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

"""Eval Config"""
C.EVAL = edict()
C.EVAL.EVAL_PRINT_FREQ = 30


""" Test """
C.TEST = edict()
# Whether to use center crop when testing
C.TEST.CROP = True

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.SYSTEM.root_dir))

"""SAVE Config"""
C.WRITE = edict()
C.WRITE.log_dir = "./results/logs"
C.WRITE.checkpoint_dir = "./results/saved_models"

# -----------------------------------------------------------------------------
# Miscellanous
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
C.AMP_OPT_LEVEL = '00'
# Path to output folder, overwritten by command line argument
C.OUTPUT = './output'
# Tag of experiment, overwritten by command line argument
C.TAG = 'default'
# Frequency to save checkpoint
C.SAVE_FREQ = 1
# Frequency to logging info
C.PRINT_FREQ = 100
# Fixed random seed
C.SEED = 0
# Perform evaluation only, overwritten by command line argument
C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
C.THROUGHPUT_MODE = False
# Debug only so that skip dataloader initialization, overwritten by command line argument
C.DEBUG_MODE = False
# local rank for DistributedDataParallel, given by command line argument
C.LOCAL_RANK = 0

### Might need it for multiscale evaluation
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.WRITE.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.WRITE.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.WRITE.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
