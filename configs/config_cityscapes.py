import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C


"""System Config"""   
C.SYSTEM = edict()
C.SYSTEM.seed = 12345
remoteip = os.popen('pwd').read()
C.SYSTEM.root_dir = os.getcwd()
C.SYSTEM.abs_dir = osp.realpath(".")
#C.SYSTEM.device_ids = [0] # for mahdi (lab-pc)
# C.SYSTEM.device_ids = [0, 1] # for mahdi (newton)
C.SYSTEM.device_ids = [0, 1, 2, 3] # for FICS


"""Dataset Config"""
C.DATASET = edict()
C.DATASET.name = 'cityscapes'
C.DATASET.root = osp.join(C.SYSTEM.root_dir, 'data/Cityscapes')
C.DATASET.mode = 'RGB'
# C.DATASET.dataset_config_path = osp.join(C.SYSTEM.root_dir, 'dataloader/cityscapes_rgbd_config.yaml')
C.DATASET.gt_transform = True
C.DATASET.num_train_imgs = 2975
C.DATASET.num_classes = 19  #for cityscape
C.DATASET.norm_mean = np.array([0.291,  0.329,  0.291]) # For CityScape
C.DATASET.norm_std = np.array([0.190,  0.190,  0.185])
C.DATASET.base_size = 512
C.DATASET.crop_size = 1024
C.DATASET.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                        'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                        'motorcycle', 'bicycle']
C.DATASET.annotation_type = "semantic"  #[instance, bbox]
C.DATASET.synthetic = False
C.DATASET.gt_mode = 'gtFine'
C.DATASET.depth_dir = 'disparity'
C.DATASET.scramble_labels = False
C.DATASET.normalize_only = False
C.DATASET.no_transforms = False
C.DATASET.power_transform = False
C.DATASET.pt_lambda = -0.5

""" Dataset --> Darken Augmentation """
C.DATASET.DARKEN = edict()
C.DATASET.DARKEN.darken = True          ## It was set False. Why?
C.DATASET.DARKEN.gamma = 2.0
C.DATASET.DARKEN.gain = 0.5
C.DATASET.DARKEN.gaussian_sigma = 0.01
C.DATASET.DARKEN.poisson = True



"""Image Config"""
C.IMAGE = edict()
C.IMAGE.background = 255
C.IMAGE.image_height = 1024
C.IMAGE.image_width = 1024



""" Model Config"""
C.MODEL = edict()
C.MODEL.backbone = 'mit_b2'
C.MODEL.pretrained_model = osp.join(C.SYSTEM.root_dir, 'pretrained/mit_b2_imagenet.pth')
C.MODEL.decoder = 'MLPDecoder'
C.MODEL.decoder_embed_dim = 512
C.MODEL.checkpoint_start_epoch = 250
C.MODEL.checkpoint_step = 5


"""Train Config"""
C.TRAIN = edict()
C.TRAIN.lr = 1e-4
C.TRAIN.lr_power = 1
C.TRAIN.optimizer = 'AdamW'
C.TRAIN.momentum = 0.9
C.TRAIN.weight_decay = 0.01

C.TRAIN.batch_size = 2
C.TRAIN.nepochs = 431
C.TRAIN.niters_per_epoch = C.DATASET.num_train_imgs // C.TRAIN.batch_size  + 1
C.TRAIN.num_workers = 4

C.TRAIN.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.TRAIN.warm_up_epoch = 10

C.TRAIN.fix_bias = True
C.TRAIN.bn_eps = 1e-3
C.TRAIN.bn_momentum = 0.1
C.TRAIN.train_print_stats = 50
C.TRAIN.resume_train = False 
C.TRAIN.resume_model_path = osp.join(C.SYSTEM.root_dir, 'Results/saved_models/07-10-23_2314/model_330.pth')


"""Eval Config"""
C.EVAL = edict()
C.EVAL.val_print_stats = 100
C.EVAL.eval_iter = 25
C.EVAL.eval_stride_rate = 2 / 3
C.EVAL.eval_scale_array = [0.75, 1, 1.25]
C.EVAL.eval_flip = False # True # 
C.EVAL.eval_crop_size = [1024, 1024]


"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.SYSTEM.root_dir))

"""SAVE Config"""
C.WRITE = edict()
C.WRITE.log_dir = "./results/logs"
C.WRITE.checkpoint_dir = "./results/saved_models"

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
