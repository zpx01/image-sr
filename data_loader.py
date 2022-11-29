import numpy as np
from imresize import imresize
import math
from data_loader_utils import *
from test_time_config import Config
import torch.utils.data as data
import utils.utils_image as util

"""
A DataLoader class for test-time training.
This class generates augmentations of test images
during test-time so that you can train/fine-tune your
model in real time.
"""
class DataLoaderPretrained(data.Dataset):
    def __init__(self, input_img, conf=Config(), sf=2, kernel=None):
        super(DataLoaderPretrained, self).__init__()
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
        for _ in range(index):
            hr_father = self.generate_hr_father()
            lr_son = self.father_to_son(hr_father)
            hr_father = np.transpose(np.expand_dims(hr_father, axis=0), (0, 3, 1, 2))
            lr_son = np.transpose(np.expand_dims(lr_son, axis=0), (0, 3, 1, 2))
            self.hr.append(hr_father)
            self.lr.append(lr_son)
    
    def __getitem__(self, index):
        # generate some image pairs
        self.generate_pairs(index)
        return self.lr[index], self.hr[index]
    
    def __len__(self):
        return len(self.hr_father_sources)






    
    
    
    

    