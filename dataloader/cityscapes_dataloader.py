'''
Source: https://github.com/crmauceri/dataset_loaders/blob/master/dataloaders/datasets/

'''
import os
import random
import numpy as np
import scipy.misc as m
from skimage import measure
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.cityscapes_dataloader_utils.SampleLoader import SampleLoader

class CityscapesDataset(data.Dataset):
    def __init__(self, cfg, split="train"):

        self.root = cfg.DATASET.root
        self.split = split
        self.cfg = cfg

        self.mode = cfg.DATASET.mode 
        self.loader = CityscapesSampleLoader(cfg, split)
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        if split == "val" or split == "test":
            self.annotations_base = os.path.join(self.root, 'gtFine', self.split)
        else:
            self.annotations_base = os.path.join(self.root, cfg.DATASET.gt_mode, self.split)


        self.depth_base = os.path.join(self.root, cfg.DATASET.depth_dir, self.split)  # {}{}'.format(split, year))


        # 'troisdorf_000000_000073' is corrupted
        self.files[split] = [x for x in self.recursive_glob(rootdir=self.images_base, suffix='.png') if 'troisdorf_000000_000073' not in x]
        self.files[split] = self.files[split][:100]
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

        self.class_names =  cfg.DATASET.class_names

    def __len__(self):
        return len(self.files[self.split])

    def get_length(self):
            return self.__len__()
    
    def __getitem__(self, index):
        img_path, depth_path, lbl_path = self.get_path(index, self.cfg.DATASET.scramble_labels)
        sample = self.loader.load_sample(img_path, depth_path, lbl_path)
        sample['id'] = img_path
        return sample

    def get_path(self, index, scramble_labels=False):
        img_path = self.files[self.split][index].rstrip()
        depth_path = os.path.join(self.depth_base,
                                  img_path.split(os.sep)[-2],
                                  os.path.basename(img_path)[:-15] + '{}.png'.format(
                                      self.cfg.DATASET.depth_dir))

        gt_mode = 'gtFine' if self.split == 'val' else self.cfg.DATASET.gt_mode
        if scramble_labels:
            r_index = random.randrange(0, len(self.files[self.split]))
            base_path = self.files[self.split][r_index].rstrip()
        else:
            base_path = img_path

        if self.cfg.DATASET.annotation_type == 'semantic':
            label_format = '{}_labelIds.png'
        elif self.cfg.DATASET.annotation_type in ['instance', 'bbox']:
            label_format = '{}_instanceIds.png'
        else:
            raise ValueError("DATASET.ANNOTATION_TYPE={} not implemented".format(self.cfg.DATASET.annotation_type))

        lbl_path = os.path.join(self.annotations_base,
            base_path.split(os.sep)[-2],
            os.path.basename(base_path)[:-15] + label_format.format(gt_mode))

        return img_path, depth_path, lbl_path

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]


class CityscapesSampleLoader(SampleLoader):
    def __init__(self, cfg, split="train"):
        super().__init__(cfg, mode=cfg.DATASET.mode, split=split,
                        base_size=cfg.DATASET.base_size, crop_size=cfg.DATASET.crop_size)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, \
                            17, 19, 20, 21, 22, \
                            23, 24, 25, 26, 27, 28, 31, \
                              32, 33]
        self.class_names = cfg.DATASET.class_names
        
        
        self.NUM_CLASSES = len(self.valid_classes)

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        # print(f'class mapping: {self.class_map}')
        # self.mode = mode
        self.normalizationFactors()

    def normalizationFactors(self):
        if self.mode == "RGBD":
            print('Using RGB-D input')
            # Data mean and std empirically determined from 1000 Cityscapes samples
            self.data_mean = [0.291,  0.329,  0.291,  0.126]
            self.data_std = [0.190,  0.190,  0.185,  0.179]
        elif self.mode == "RGB":
            print('Using RGB input')
            self.data_mean = [0.291,  0.329,  0.291]
            self.data_std = [0.190,  0.190,  0.185]
        elif self.mode == "RGB_HHA":
            print('Using RGB HHA input')
            self.data_mean =  [0.291,  0.329,  0.291, 0.080, 0.621, 0.370]
            self.data_std =  [0.190,  0.190,  0.185, 0.061, 0.355, 0.196]


    def getLabels(self, lbl_path):
        if self.cfg.DATASET.annotation_type == 'semantic':
            _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
            _tmp = self.encode_segmap(_tmp)
            _target = Image.fromarray(_tmp)
        elif self.cfg.DATASET.annotation_type == 'instance':
            _tmp = np.array(Image.open(lbl_path))
            _target = self.decode_instance_map(_tmp)
        elif self.cfg.DATASET.annotation_type == 'bbox':
            _tmp = np.array(Image.open(lbl_path))
            _target = self.decode_bbox(_tmp)
        else:
            raise ValueError("DATASET.ANNOTATION_TYPE={} not implemented".format(self.cfg.DATASET.ANNOTATION_TYPE))
        return _target


    def loadDepth(self, depth_path):
        if self.cfg.DATASET.synthetic:
            _depth_arr = np.array(Image.open(depth_path), dtype=int)
            # if np.max(_depth_arr) > 10000:
            #     print("Large max depth: {} {}".format(np.max(_depth_arr), depth_path))
            _depth_arr = (_depth_arr.astype('float32') / 10000.) * 256
            np.clip(_depth_arr, 0, 255, out=_depth_arr)
            _depth_arr = _depth_arr.astype(np.uint8)
            _depth = Image.fromarray(_depth_arr).convert('L')
        elif self.mode == 'RGBD':
            _disparity_arr = np.array(Image.open(depth_path)).astype(np.float32)
            # Conversion from https://github.com/mcordts/cityscapesScripts see `disparity`
            # See https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
            _disparity_arr[_disparity_arr > 0] = (_disparity_arr[_disparity_arr > 0] - 1.0) / 256.
            _depth_arr = np.zeros(_disparity_arr.shape)
            _depth_arr[_disparity_arr > 0] = 0.2 * 2262 / _disparity_arr[_disparity_arr > 0]
            # _depth_arr[_depth_arr > 60] = 60
            # _depth_arr = _depth_arr / 60. * 255.
            # _depth_arr = _depth_arr.astype(np.uint8)
            # _depth = Image.fromarray(_depth_arr).convert('L')
            _depth = Image.fromarray(_depth_arr)
        elif self.mode == 'RGB_HHA':
            # Depth channel is inverse with 25600 / depth
            # Height channel is inverse with 2560 / height
            _depth = Image.open(depth_path).convert('RGB')
        return _depth

    def encode_segmap(self, mask):
        # Put all void classes to zero
        ##### mapping all void classes to ignore_index = 255
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_instance_map(self, mask):
        # Produces a wxhxn matrix where each n is an instance mask and the values of the instance mask are the type label.
        instanceIds = np.unique(mask)
        target = np.zeros((mask.shape[0], mask.shape[1], len(instanceIds)))
        for i, instanceId in enumerate(instanceIds):
            tmp = np.zeros(mask.shape)
            if instanceId < 1000:
                tmp[mask == instanceId] = instanceId
            else:
                tmp[mask == instanceId] = instanceId / 1000
            target[:, :, i] = tmp
        return target

    def decode_bbox(self, mask):
        h, w = mask.shape
        instanceIds = np.unique(mask)
        target = np.zeros((1, 5))
        for i, instanceId in enumerate(instanceIds):
            # Find label
            if instanceId < 1000:
                label = instanceId
            else:
                label = instanceId / 1000
            # Find bbox
            sub_instances, num_sub = measure.label(mask == instanceId, return_num=True)
            for sub_instanceId in range(1, num_sub+1):
                tmp = sub_instances == sub_instanceId
                nonzero_idx = np.nonzero(tmp)
                min_y, min_x = np.min(nonzero_idx, axis=1)
                max_y, max_x = np.max(nonzero_idx, axis=1)
                bbox_h = max_y-min_y
                bbox_w = max_x-min_x
                center_x = min_x + bbox_w/2.0
                center_y = min_y + bbox_h/2.0
                #Normalize bbox
                bbox = np.array(([label, center_x/w, center_y/h, bbox_w/w, bbox_h/h],))
                target = np.concatenate((target, bbox))
        return target[1:, :]


if __name__ == '__main__':
    
    from cfg_defaults import get_cfg_defaults
    from dataloader.cityscapes_dataloader_utils.utils import decode_segmap, sample_distribution
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Test cityscapes Loader")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file) # dataloader/cityscapes_rgbd_config.yaml
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)

    cityscapes_train = CityscapesDataset(cfg, split='train')
    # cityscapes_val = CityscapesDataset(cfg, split='val')
    # cityscapes_test = CityscapesDataset(cfg, split='test')

    dataloader = DataLoader(cityscapes_train, batch_size=8, shuffle=False, num_workers=2)
    
    ## checking next(iter)
    # images, labels = next(iter(dataloader))
    # print(images.shape, labels.shape)
    # exit()
    
    # next(iter(dataloader)))
    
    for ii, sample in enumerate(dataloader):
        # print(sample["id"])
        # print(sample['image'].shape, sample['label'].shape)
        print(torch.unique(sample['label']))
        
        images = sample['image']
        labels = sample['label']

        # Plot the images and labels
        # fig, axs = plt.subplots(4, 2, figsize=(10, 20))

        # for jj in range(images.size()[0]):
        #     img = images[jj].numpy().transpose((1, 2, 0))
        #     gt = labels[jj].numpy()
        #     tmp = np.array(gt).astype(np.uint8)
        #     segmap = decode_segmap(tmp, dataset='cityscapes')
        #     axs[jj, 0].imshow(img)
        #     axs[jj, 1].imshow(segmap)

        # plt.show()
        # plt.savefig('output_cityscapes_dataloader_test.png')
        if ii>10:
            break
    # print(sample_distribution(cityscapes_train))
