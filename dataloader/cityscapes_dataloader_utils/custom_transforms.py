import torch
import random
import numpy as np
import torchvision
import skimage

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2



class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, sample):
        """Pad images according to ``self.size``."""
        img = np.array(sample['image'])

        if self.size is not None:
            width = max(self.size[1] - img.shape[1], 0)
            height = max(self.size[0] - img.shape[0], 0)
            padding = (0, 0, width, height)

        if self.size is not None:
            img = cv2.copyMakeBorder(
                            img,
                            padding[1],
                            padding[3],
                            padding[0],
                            padding[2],
                            cv2.BORDER_CONSTANT,
                            value=self.pad_val)
        return img

    def _pad_seg(self, sample):
        """Pad masks according to ``results['pad_shape']``."""
        label = np.array(sample['label'])

        if self.size is not None:
            width = max(self.size[1] - label.shape[1], 0)
            height = max(self.size[0] - label.shape[0], 0)
            padding = (0, 0, width, height)

        if self.size is not None:
            label = cv2.copyMakeBorder(
                                    label,
                                    padding[1],
                                    padding[3],
                                    padding[0],
                                    padding[2],
                                    cv2.BORDER_CONSTANT,
                                    value=self.seg_pad_val)
        return label

    def __call__(self, sample):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        img = self._pad_img(sample)
        label = self._pad_seg(sample)

        sample['image'] = Image.fromarray(img)
        sample['label'] = Image.fromarray(label)
        return sample

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str





class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, sample):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = sample['image']
        img = np.array(img)
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(np.array(sample['label']), crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        
        # crop semantic seg
        seg = self.crop(np.array(sample['label']), crop_bbox)
        
        sample['image'] = Image.fromarray(img)
        sample['label'] = Image.fromarray(seg)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'





class Resize(object):
    def __init__(self, ratio_range):
        self.img_scale = None
        self.ratio_range = ratio_range
        pass

    def _random_scale(self,):
        min_ratio, max_ratio = self.ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        # scale is a tuple of (h, w)
        scale = int(self.img_scale[0] * ratio), int(self.img_scale[1] * ratio) 
        # we make it float according to mmcv
        return scale
    
    def _resize_img(self, img, scale):
        
        h, w = img.size
        max_edge = max(h, w)
        scale_factor = max(scale) / max_edge
        new_size = int(w*float(scale_factor)+0.5), int(h*float(scale_factor)+0.5)
        rescaled_img = img.resize(new_size, Image.BILINEAR)
        return rescaled_img

    def _resize_label(self, label, scale):
        h, w = label.size
        max_edge = max(h, w)
        scale_factor = max(scale) / max_edge
        new_size = int(w*float(scale_factor)+0.5), int(h*float(scale_factor)+0.5)

        rescaled_label = label.resize(new_size, Image.NEAREST)
        
        return rescaled_label   

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        depth = sample['depth']
        
        self.img_scale = img.size
        scale = self._random_scale()
        rescaled_img = self._resize_img(img, scale)
        label = self._resize_label(label, scale)

        return {'image': rescaled_img,
                'depth': depth,
                'label': label}
    







class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from RGB to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to RGB
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def random_brightness(self, img):
        if random.randint(0, 2):
            brightness_factor = random.uniform(1 - self.brightness_delta, 1 + self.brightness_delta)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        return img

    def random_contrast(self, img):
        if random.randint(0, 2):
            contrast_factor = random.uniform(self.contrast_lower, self.contrast_upper)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        return img

    def random_saturation(self, img):
        if random.randint(0, 2):
            saturation_factor = random.uniform(self.saturation_lower, self.saturation_upper)
            img = ImageEnhance.Color(img).enhance(saturation_factor)
        return img

    def random_hue(self, img):
        if random.randint(0, 2):
            img = img.convert('HSV')
            hue = random.randint(-self.hue_delta, self.hue_delta)
            img = img.point(lambda i: (i + hue) % 256)
            img = img.convert('RGB')
        return img

    def __call__(self, results):
        img = results['image']

        # Random brightness
        img = self.random_brightness(img)

        # Mode == 0 --> do random contrast first
        # Mode == 1 --> do random contrast last
        mode = random.randint(0, 2)
        if mode == 1:
            img = self.random_contrast(img)

        # Random saturation
        img = self.random_saturation(img)

        # Random hue
        img = self.random_hue(img)

        # Random contrast
        if mode == 0:
            img = self.random_contrast(img)

        results['image'] = img
        return results


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0., 0.), std=(1., 1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        img = sample['image']
        img = np.array(img).astype(np.float32)

        img /= 255.0
        img -= self.mean[:3]
        img /= self.std[:3]

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)

        depth = sample['depth']
        if not isinstance(depth, list):
            depth = np.array(depth).astype(np.float32)
            depth /= 255.0
            depth -= self.mean[3:]
            depth /= self.std[3:]

        return {'image': img,
                'depth': depth,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        mask = sample['label']
        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        depth = sample['depth']
        if not isinstance(depth, list):
            depth = np.array(depth).astype(np.float32)
            if len(depth.shape) == 3:
                depth = depth.transpose((2, 0, 1))
            depth = torch.from_numpy(depth).float()

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        coin_flip = random.random()
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if coin_flip < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if not isinstance(depth, list):
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1*self.degree, self.degree)

        img = sample['image']
        img = img.rotate(rotate_degree, Image.BILINEAR)

        mask = sample['label']
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        depth = sample['depth']
        if not isinstance(depth, list):
             depth = depth.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        coin_flip = random.random()
        rand_radius = random.random()
        mask = sample['label']

        img = sample['image']
        depth = sample['depth']
        if coin_flip < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=rand_radius))
            if not isinstance(depth, list):
                depth = depth.filter(ImageFilter.GaussianBlur(
                radius=rand_radius))

        return {'image': img,
                'depth': depth,
                'label': mask}
# class RandomResizeCrop(object):
#     def __init__(self, img_size, scale, ratio):
#         self.img_size = img_size
#         self.scale = scale
#         self.ratio = ratio
#         self.crop_transform = transforms.RandomResizedCrop(size=(300, 600))

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         depth = sample['depth']




class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        #[256, 1024] with base size 512
        #[512, 2048] with base size 1024
        w, h = img.size

        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size     
            ow = int(1.0 * w * oh / h)  
        mask = mask.resize((ow, oh), Image.NEAREST) 
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w_resize, h_resize = mask.size      
        x1 = random.randint(0, w_resize - self.crop_size) 
        y1 = random.randint(0, h_resize - self.crop_size)   

        
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = img.resize((ow, oh), Image.BILINEAR)
        if short_size < self.crop_size:
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if not isinstance(depth, list):
            depth = depth.resize((ow, oh), Image.BILINEAR)
            if short_size < self.crop_size:
                depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)
            depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        # mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w_resize, h_resize = mask.size
        x1 = int(round((w_resize - self.crop_size) / 2.))
        y1 = int(round((h_resize - self.crop_size) / 2.))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        img = img.resize((ow, oh), Image.BILINEAR)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # if not isinstance(depth, list):
        #     depth = depth.resize((ow, oh), Image.BILINEAR)
        #     depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']

        img = sample['image']
        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)

        mask = mask.resize(self.size, Image.NEAREST)

        if not isinstance(depth, list):
            depth = depth.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'depth': depth,
                'label': mask}

# Based on method from
#  Structure-Revealing Low-Light Image EnhancementVia Robust Retinex Model.
#          Li et al. Transactions on Image Processing, 2018.
# "We synthesize low-light images by  first  applying Gamma correction(with?=2.2)  (...)
#  and  then  adding Poisson noise and white Gaussian noise to Gamma corrected images.
#  In our work, we use the built-in function of MATLAB imnoise to generate Poisson noise.
#  For Gaussian noise, we use s=5  to  simulate  the  noise  level  in  most  natural  low-light images."
class RandomDarken(object):
    def __init__(self, cfg, darken):
        self.darken = darken
        self.gaussian_var = cfg.DATASET.DARKEN.gaussian_sigma*cfg.DATASET.DARKEN.gaussian_sigma
        self.poisson = cfg.DATASET.DARKEN.poisson

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if self.darken:
            # Darken Image
            gamma = 1.0 + random.random() * 1.2
            gain = 1 - random.random() / 2.0
            img = torchvision.transforms.functional.adjust_gamma(img, gamma, gain)

            # Add noise
            img_arr = np.array(img).astype(np.float32) / 255.

            # Shot noise, proportional to number of photons measured
            if self.poisson:
                img_arr = skimage.util.random_noise(img_arr, mode='poisson', clip=True)
            # Temperature noise, constant for sensor at temperature
            if self.gaussian_var > 0:
                img_arr = skimage.util.random_noise(img_arr, mode='gaussian', var=self.gaussian_var, clip=True)
            img = Image.fromarray(np.uint8(img_arr * 255.))

        return {'image': img,
                'depth': depth,
                'label': mask}

class Darken(object):
    def __init__(self, cfg):
        #, gamma=2.0, gain=0.5, gaussian_m = 5./255.
        self.darken = cfg.DATASET.DARKEN.DARKEN  # size: (h, w)
        self.gamma = cfg.DATASET.DARKEN.GAMMA
        self.gain = cfg.DATASET.DARKEN.GAIN
        self.gaussian_var = cfg.DATASET.DARKEN.GAUSSIAN_SIGMA * cfg.DATASET.DARKEN.GAUSSIAN_SIGMA
        self.poisson = cfg.DATASET.DARKEN.POISSON

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if self.darken:
            # Darken Image
            img = torchvision.transforms.functional.adjust_gamma(img, self.gamma, self.gain)

            img_arr = np.array(img).astype(np.float32) / 255.

            # Add noise
            # Shot noise, proportional to number of photons measured
            if self.poisson:
                img_arr = skimage.util.random_noise(img_arr, mode='poisson', clip=True)
            # Temperature noise, constant for sensor at temperature
            if self.gaussian_var > 0:
                img_arr = skimage.util.random_noise(img_arr, mode='gaussian', var=self.gaussian_var, clip=True)

            img = Image.fromarray(np.uint8(img_arr * 255.))

        return {'image': img,
                'depth': depth,
                'label': mask}

## Reverses gamma correction and gain to show the effect of adding noise to the image. Noise is not-reversed
class UnDarken(object):
    def __init__(self, cfg):
        #, gamma=2.0, gain=0.5, gaussian_m = 5./255.
        self.darken = cfg.DATASET.DARKEN.DARKEN  # size: (h, w)
        self.gamma = 1.0 / cfg.DATASET.DARKEN.GAMMA # To reverse gamma correction, take the gamma root

        if cfg.DATASET.DARKEN.GAIN == 0:
            self.gain = 0.0 #No way to reverse
        else:
            self.gain = 1.0 / cfg.DATASET.DARKEN.GAIN # To reverse gain, multiply by the inverse

    def __call__(self, sample):
        mask = sample['label']
        depth = sample['depth']
        img = sample['image']

        if self.darken:
            # Darken Image
            img = torchvision.transforms.functional.adjust_gamma(img, self.gamma, self.gain)
            img = Image.fromarray(np.uint8(img))

        return {'image': img,
                'depth': depth,
                'label': mask}