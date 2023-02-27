import numpy as np
from imresize import imresize
import math
import torch
from data_loader_utils import *
from test_time_config import Config
import torch.utils.data as data
import utils.utils_image as util
from torchvision.transforms import RandomCrop, ToTensor, CenterCrop
import torchvision.transforms as T
import glob
import tqdm
import PIL.Image
import random
from utils.utils_calculate_psnr_ssim import to_y_channel


class DataLoaderPretrained(data.Dataset):
    """
    A DataLoader class for test-time training.
    This class generates augmentations of test images
    during test-time so that you can train/fine-tune your
    model in real time.
    """
    def __init__(self, input_img, conf=None, sf=2, kernel=None):
        """sf: the scale factor."""
        super(DataLoaderPretrained, self).__init__()
        if conf is None: conf = Config()
        self.input_img = input_img
        self.hr_father_sources = [self.input_img]
        self.sf = sf
        self.kernel = None
        self.conf = conf
        self.num_workers=8

    def generate_hr_father(self):
        return random_augment(ims=self.hr_father_sources,
                              base_scales=[1.0] + self.conf.scale_factors,
                              leave_as_is_probability=1,
                              no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                              min_scale=self.conf.augment_min_scale,
                              max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_father_sources)-1],
                              allow_rotation=self.conf.augment_allow_rotation,
                              scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                              shear_sigma=self.conf.augment_shear_sigma,
                              crop_size=self.conf.crop_size)
    
    def father_to_son(self, hr_father):
        return imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
    
    def generate_pairs(self, index):
        self.hr = []
        self.lr = []
        for _ in range(index+1):
            hr_father = self.generate_hr_father()
            lr_son = self.father_to_son(hr_father)
            hr_father = np.transpose(np.expand_dims(hr_father, axis=0), (0, 3, 1, 2))
            lr_son = np.transpose(np.expand_dims(lr_son, axis=0), (0, 3, 1, 2))
            hr_father, lr_son = hr_father.squeeze(), lr_son.squeeze()
            self.hr.append(torch.Tensor(hr_father))
            self.lr.append(torch.Tensor(lr_son))
    
    def __getitem__(self, index):
        # generate some image pairs
        self.generate_pairs(index)
        return self.lr[index], self.hr[index]
    
    def __len__(self):
        return len(self.hr_father_sources)  


class DataLoaderMerger(data.Dataset):
    """
    A DataLoader class for test-time training.
    This class generates augmentations of test images
    during test-time so that you can train/fine-tune your
    model in real time.
    """
    def __init__(self, hr_path, orig_path, ttt_path, split='train', img_size: int = 48):
        super(DataLoaderClassification, self).__init__()
        if split == 'train':
            self.random_crop = RandomCrop(img_size)
        else:
            assert split == 'test'
            self.random_crop = CenterCrop(img_size)
        self.to_tensor = ToTensor()
        self._hr_paths = sorted(list(glob.glob(hr_path + '/*.png')))
        self.orig_paths = []
        self.split = split
        self.img_size = img_size
        self.ttt_paths = []
        self.hr_paths = []
        for path in self._hr_paths:
            name = os.path.split(path)[-1].replace('.png', '')
            ttt_image_name = f'{name}.png'
            orig_image_name = ttt_image_name
            # Do it once to eliminate cases that don't provide inforamtion:
            full_orig_path = os.path.join(orig_path, orig_image_name)
            full_ttt_path = os.path.join(ttt_path, ttt_image_name)
            self.orig_paths.append(full_orig_path)
            self.ttt_paths.append(full_ttt_path)
            self.hr_paths.append(path)

    def _crop_according_to_smallest(self, image1, image2, image3, image4, scale=4):
        # (FIXED) Added appropriate cropping for GT
        images = [np.array(x) for x in (image1, image2, image3, image4)]
        images[3] = np.transpose(images[3] if images[3].shape[2] == 1 else images[3][:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        images[3] = torch.from_numpy(images[3]).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        _, _, h_old, w_old = images[3].size()
        images[2] = images[2][:h_old * scale, :w_old * scale, ...]  # crop gt
        images[2] = np.squeeze(images[2])
        height = min([x.shape[0] for x in images])
        width = min([x.shape[1] for x in images])
        return [PIL.Image.fromarray(x[:height, :width]) for x in images]

    def __getitem__(self, index):
        # generate some image pair
        image_orig = PIL.Image.open(self.orig_paths[index])
        image_ttt = PIL.Image.open(self.ttt_paths[index])
        image_hr =  PIL.Image.open(self.hr_paths[index])
        image_lr = PIL.Image.open(self.lr_paths[index])
        (image_orig, 
         image_ttt, 
         image_hr, 
         image_lr) = self._crop_according_to_smallest(image_orig, image_ttt, image_hr, image_lr, scale=4)
        
        image_orig = self.to_tensor(image_orig)
        image_ttt = self.to_tensor(image_ttt)
        image_hr = self.to_tensor(image_hr)
        if self.split == 'train':
            params = self.random_crop.get_params(image_orig, (self.img_size, self.img_size))
            image_orig = T.functional.crop(image_orig, *params)
            image_ttt = T.functional.crop(image_ttt, *params)
            image_hr = T.functional.crop(image_hr, *params)
        else:
            image_orig = self.random_crop(image_orig)
            image_ttt = self.random_crop(image_ttt)
            image_hr = self.random_crop(image_hr)
        return image_orig, image_ttt, image_hr

    def __len__(self):
        return len(self.hr_paths)


class DataLoaderClassification(data.Dataset):
    """
    A DataLoader class for test-time training.
    This class generates augmentations of test images
    during test-time so that you can train/fine-tune your
    model in real time.
    """
    def __init__(self, hr_path, lr_path, orig_path, ttt_path, threshold, 
                 initial_signal_threshold: int = 200, split='train', img_size: int = 48):
        super(DataLoaderClassification, self).__init__()
        if split == 'train':
            self.random_crop = RandomCrop(img_size)
        else:
            assert split == 'test'
            self.random_crop = CenterCrop(img_size)
        self.to_tensor = ToTensor()
        self.threshold = threshold
        self.img_size = img_size
        self.initial_signal_threshold = initial_signal_threshold
        self._hr_paths = sorted(list(glob.glob(hr_path + '/*.png')))
        self._lr_paths = sorted(list(glob.glob(lr_path + '/*.png')))
        self.orig_paths = []
        self.split = split
        self.ttt_paths = []
        self.hr_paths = []
        self.lr_paths = []
        for idx, path in enumerate(self._hr_paths):
            name = os.path.split(path)[-1].replace('.png', '')
            ttt_image_name = f'{name}.png'
            orig_image_name = ttt_image_name
            # Do it once to eliminate cases that don't provide inforamtion:
            full_orig_path = os.path.join(orig_path, orig_image_name)
            full_ttt_path = os.path.join(ttt_path, ttt_image_name)
            self.orig_paths.append(full_orig_path)
            self.ttt_paths.append(full_ttt_path)
            self.hr_paths.append(path)
            self.lr_paths.append(self._lr_paths[idx])

    def create_masks(self, hr_image: np.array, orig_image: np.array, ttt_image: np.array):
        hr_image = to_y_channel(hr_image)
        ttt_image = to_y_channel(ttt_image)
        orig_image = to_y_channel(orig_image)
        distance_ttt = np.mean((hr_image - ttt_image) ** 2, axis=2)
        distance_orig = np.mean((hr_image - orig_image) ** 2, axis=2)
        mask = np.zeros((hr_image.shape[0], hr_image.shape[1]))
        signal_mask = np.zeros((hr_image.shape[0], hr_image.shape[1]))
        mask[distance_ttt < distance_orig - self.threshold] = 1.
        signal_mask[distance_ttt < distance_orig - self.threshold] = 1.
        mask[distance_orig < distance_ttt - self.threshold] = 0
        signal_mask[distance_orig < distance_ttt - self.threshold] = 1.
        # Balance masks:
        if np.sum(signal_mask * mask) == 0 or np.sum(signal_mask *(1- mask)) == 0:
            return np.zeros_like(mask), np.zeros_like(signal_mask)
        mask *= signal_mask
        balance = np.sum(signal_mask * mask) / np.sum(signal_mask *(1- mask))
        if balance > 1:
            signal_mask = (signal_mask * (1 - mask) + 
                           signal_mask * mask * (np.random.uniform(0, 1, size=mask.shape) < (1 / balance)))
        else:
            signal_mask = (signal_mask * mask + 
                           signal_mask * (1 - mask) * (np.random.uniform(0, 1, size=mask.shape) < balance))
        mask *= signal_mask
        return mask * 255, signal_mask * 255

    def _crop_according_to_smallest(self, image1, image2, image3, image4, scale=4):
        # (FIXED) Added appropriate cropping for GT
        images = [np.array(x) for x in (image1, image2, image3, image4)]
        images[3] = np.transpose(images[3] if images[3].shape[2] == 1 else images[3][:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        images[3] = torch.from_numpy(images[3]).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        _, _, h_old, w_old = images[3].size()
        images[2] = images[2][:h_old * scale, :w_old * scale, ...]  # crop gt
        images[2] = np.squeeze(images[2])
        height = min([x.shape[0] for x in images])
        width = min([x.shape[1] for x in images])
        return [PIL.Image.fromarray(x[:height, :width]) for x in images]
        
    def __getitem__(self, index):
        # generate some image pair
        image_orig = PIL.Image.open(self.orig_paths[index])
        image_ttt = PIL.Image.open(self.ttt_paths[index])
        image_hr =  PIL.Image.open(self.hr_paths[index])
        image_lr = PIL.Image.open(self.lr_paths[index])
        (image_orig, 
         image_ttt, 
         image_hr, 
         image_lr) = self._crop_according_to_smallest(image_orig, image_ttt, image_hr, image_lr, scale=4)
        try:
            mask, signal_mask = self.create_masks(np.array(image_hr), np.array(image_orig), np.array(image_ttt))
        except ValueError as e:
            print(self.orig_paths[index], self.ttt_paths[index], self.hr_paths[index])
            print(e)
            raise e
        mask = PIL.Image.fromarray(np.uint8(mask))
        signal_mask = PIL.Image.fromarray(np.uint8(signal_mask)) 
        
        image_orig = self.to_tensor(image_orig)
        mask = self.to_tensor(mask)
        signal_mask = self.to_tensor(signal_mask)
        image_ttt = self.to_tensor(image_ttt)
        if self.split == 'train':
            params = self.random_crop.get_params(image_orig, (self.img_size, self.img_size))
            signal_mask = T.functional.crop(signal_mask, *params)
        else:
            signal_mask = self.random_crop(signal_mask)
        tries = 0
        while torch.sum(signal_mask) < self.initial_signal_threshold:
            # While we don't have enought signal.
            tries += 1
            if tries == 10: 
                tries = 0
                index = random.randint(0, len(self.hr_paths)-1)
                image_orig = PIL.Image.open(self.orig_paths[index])
                image_ttt = PIL.Image.open(self.ttt_paths[index])
                image_hr =  PIL.Image.open(self.hr_paths[index])
                image_lr = PIL.Image.open(self.lr_paths[index])
                (image_orig, 
                image_ttt, 
                image_hr, 
                image_lr) = self._crop_according_to_smallest(image_orig, image_ttt, image_hr, image_lr, scale=4)
                mask, signal_mask = self.create_masks(np.array(image_hr), np.array(image_orig), np.array(image_ttt))
                mask = PIL.Image.fromarray(np.uint8(mask))
                signal_mask = PIL.Image.fromarray(np.uint8(signal_mask)) 
                image_orig = self.to_tensor(image_orig)
                mask = self.to_tensor(mask)
                signal_mask = self.to_tensor(signal_mask)
                image_ttt = self.to_tensor(image_ttt)
            if self.split == 'train':
                params = self.random_crop.get_params(image_orig, (48, 48))
                signal_mask = T.functional.crop(signal_mask, *params)
            else:
                signal_mask = self.random_crop(signal_mask) 
        if self.split == 'train':
            image_orig = T.functional.crop(image_orig, *params)
            image_ttt = T.functional.crop(image_ttt, *params)
            mask = T.functional.crop(mask, *params)
        else:
            image_orig = self.random_crop(image_orig)
            image_ttt = self.random_crop(image_ttt)
            mask = self.random_crop(mask)
        return image_orig, image_ttt, mask, signal_mask

    def __len__(self):
        return len(self.hr_paths)
