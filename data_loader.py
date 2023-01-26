import numpy as np
from imresize import imresize
import math
import torch
from data_loader_utils import *
from test_time_config import Config
import torch.utils.data as data
import utils.utils_image as util
from torchvision.transforms import RandomCrop, ToTensor
import torchvision.transforms as T
import glob
import tqdm
import PIL.Image
import random

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
            self.hr.append(hr_father)
            self.lr.append(lr_son)
    
    def __getitem__(self, index):
        # generate some image pairs
        self.generate_pairs(index)
        return self.lr[index], self.hr[index]
    
    def __len__(self):
        return len(self.hr_father_sources)




class DataLoaderClassification(data.Dataset):
    """
    A DataLoader class for test-time training.
    This class generates augmentations of test images
    during test-time so that you can train/fine-tune your
    model in real time.
    """
    def __init__(self, hr_path, orig_path, ttt_path, threshold, initial_signal_threshold: int = 200):
        super(DataLoaderClassification, self).__init__()
        self.random_crop = RandomCrop(128)
        self.to_tensor = ToTensor()
        self.threshold = threshold
        self.initial_signal_threshold = initial_signal_threshold
        self._hr_paths = sorted(list(glob.glob(hr_path + '/*.png')))
        self.orig_paths = []
        self.ttt_paths = []
        self.hr_paths = []
        for path in self._hr_paths:
            name = os.path.split(path)[-1].replace('.png', '')
            ttt_image_name = f'{name}_1_24_23_ttt_ttt.png'
            orig_image_name = ttt_image_name.replace('ttt', 'orig')
            # Do it once to eliminate cases that don't provide inforamtion:
            full_orig_path = os.path.join(orig_path, orig_image_name)
            full_ttt_path = os.path.join(ttt_path, ttt_image_name)
            self.orig_paths.append(full_orig_path)
            self.ttt_paths.append(full_ttt_path)
            self.hr_paths.append(path)

    def create_masks(self, hr_image: np.array, orig_image: np.array, ttt_image: np.array):
        distance_ttt = np.mean((hr_image - ttt_image) ** 2, axis=2)
        distance_orig = np.mean((hr_image - orig_image) ** 2, axis=2)
        mask = np.zeros((hr_image.shape[0], hr_image.shape[1]))
        signal_mask = np.zeros((hr_image.shape[0], hr_image.shape[1]))
        mask[distance_ttt < distance_orig - self.threshold] = 255.
        signal_mask[distance_ttt < distance_orig - self.threshold] = 255.
        mask[distance_orig < distance_ttt - self.threshold] = 0
        signal_mask[distance_orig < distance_ttt - self.threshold] = 255.
        return mask, signal_mask

    def __getitem__(self, index):
        # generate some image pair
        image_orig = PIL.Image.open(self.orig_paths[index])
        image_ttt = PIL.Image.open(self.ttt_paths[index])
        image_hr =  PIL.Image.open(self.hr_paths[index])
        mask, signal_mask = self.create_masks(np.array(image_hr), np.array(image_orig), np.array(image_ttt))
        mask = PIL.Image.fromarray(np.uint8(mask))
        signal_mask = PIL.Image.fromarray(np.uint8(signal_mask)) 
        
        image_orig = self.to_tensor(image_orig)
        mask = self.to_tensor(mask)
        signal_mask = self.to_tensor(signal_mask)
        image_ttt = self.to_tensor(image_ttt)
        params = self.random_crop.get_params(image_orig, (48, 48))
        signal_mask = T.functional.crop(signal_mask, *params)
        tries = 0
        while torch.sum(signal_mask) < self.initial_signal_threshold:
            tries += 1
            if tries == 10: 
                tries = 0
                index = random.randint(0, len(self.hr_paths)-1)
                image_orig = PIL.Image.open(self.orig_paths[index])
                image_ttt = PIL.Image.open(self.ttt_paths[index])
                image_hr =  PIL.Image.open(self.hr_paths[index])
                mask, signal_mask = self.create_masks(np.array(image_hr), np.array(image_orig), np.array(image_ttt))
                mask = PIL.Image.fromarray(np.uint8(mask))
                signal_mask = PIL.Image.fromarray(np.uint8(signal_mask)) 
                image_orig = self.to_tensor(image_orig)
                mask = self.to_tensor(mask)
                signal_mask = self.to_tensor(signal_mask)
                image_ttt = self.to_tensor(image_ttt)

            params = self.random_crop.get_params(image_orig, (48, 48))
            signal_mask = T.functional.crop(signal_mask, *params)
        image_orig = T.functional.crop(image_orig, *params)
        image_ttt = T.functional.crop(image_ttt, *params)
        mask = T.functional.crop(mask, *params)
        return image_orig, image_ttt, mask, signal_mask

    def __len__(self):
        return len(self.hr_paths)


    
    
    
    

    