import os
import random
import math
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from copy import deepcopy


_IMAGE_EXT = ('png', 'jpg', 'jpeg')


def convert_sequential(img):
    u = torch.unique(img)  # unique values
    img2 = torch.zeros_like(img)  # construct new image
    r = torch.arange(len(u), dtype=torch.uint8)  # new values
    for i,k in enumerate(u): img2[torch.where(img==k)] = r[i]  # convert
    return img2


class CustomTransform(object):
    def __init__(
            self,size=256,
            angle=10,
            translate=0.1,
            scale=0.1,
            shear=10,
            b_factor=0.3,
            c_factor=0.3,
            hflip=0.5,
            mean=0.5,
            std=0.5,
        ):
        self.size = size
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.b_factor = b_factor
        self.c_factor = c_factor
        self.hflip = hflip
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, mask = data

        size = self.size
        angle = 2 * self.angle * random.random() - self.angle # angle in [-self.angle,self.angle]
        translate = (self.translate * random.random(), self.translate * random.random())
        scale = 2 * self.scale * random.random() + 1 - self.scale # scale in [1-self.scale,1+self.scale]
        shear = 2 * self.shear * random.random() - self.shear # shear in [-self.shear,self.shear]
        b_factor = 2 * self.b_factor * random.random() + 1 - self.b_factor # b_factor in [1-self.b_factor,1+self.b_factor]
        c_factor = 2 * self.c_factor * random.random() + 1 - self.c_factor # c_factor in [1-self.c_factor,1+self.c_factor]
        hflip = False
        if random.random() < self.hflip:
            hflip = True
        mean = self.mean
        std = self.std

        #image = self.totensor(image)
        image = transforms.functional.resize(image, (size, size), transforms.InterpolationMode.BICUBIC)
        image = transforms.functional.affine(image, angle, translate, scale, shear)
        image = transforms.functional.adjust_brightness(image, b_factor)
        image = transforms.functional.adjust_contrast(image, c_factor)
        #if hflip:
        #    image = transforms.functional.hflip(image)
        image = image.float() / 255
        image = transforms.functional.normalize(image, [mean], [std])

        #mask = self.totensor(mask)
        mask = transforms.functional.resize(mask, (size, size), transforms.InterpolationMode.NEAREST)
        mask = transforms.functional.affine(mask, angle, translate, scale, shear)
        #if hflip:
        #    mask = transforms.functional.hflip(mask)
        # mask = mask.float() / 85 # only {0,85,170,255} values are in mask
        mask = convert_sequential(mask)
        mask = mask.long()

        return image, mask


class SegDataset(Dataset):
    def __init__(
            self,
            image_dir,
            mask_dir,
            aug_image_dir=None,
            aug_mask_dir=None,
            aug_prop=1,
            transform=None,
            device='cpu'
        ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.aug_image_dir = aug_image_dir
        self.aug_mask_dir = aug_mask_dir
        self.aug_prop = aug_prop
        self.aug_len = 0

        self.device = device
        self.transform = transform

        frames_names = sorted(os.listdir(self.image_dir))
        masks_names = sorted(os.listdir(self.mask_dir))
        self.frames = [os.path.join(self.image_dir, name) for name in frames_names]
        self.masks = [os.path.join(self.mask_dir, name) for name in masks_names]
        
        if self.aug_image_dir and self.aug_mask_dir:
            frames_names = sorted(os.listdir(self.aug_image_dir))
            masks_names = sorted(os.listdir(self.aug_mask_dir))
            self.aug_len = math.floor(self.aug_prop * len(frames_names))
            paired_names = list(zip(frames_names, masks_names))
            random.shuffle(paired_names)
            paired_names = paired_names[:self.aug_len]
            frames_names, masks_names = [], []
            for f_name, m_name in paired_names:
                frames_names.append(f_name)
                masks_names.append(m_name)
            frames_names = [os.path.join(self.aug_image_dir, name) for name in frames_names]
            masks_names = [os.path.join(self.aug_mask_dir, name) for name in masks_names]
            self.frames.extend(frames_names)
            self.masks.extend(masks_names)

        self.frames = self._remove_non_image(self.frames)
        self.masks = self._remove_non_image(self.masks)
        
    def __len__(self):
        return len(self.frames)
    
    def _remove_non_image(self, name_list):
        final_list = []
        for name in name_list:
            for ext in _IMAGE_EXT:
                if name.endswith(ext):
                    final_list.append(name)
                    break
        return final_list

    def __getitem__(self, index):
        image = read_image(self.frames[index], mode=ImageReadMode.GRAY)
        mask = read_image(self.masks[index], mode=ImageReadMode.GRAY)

        if self.transform:
            image, mask = self.transform((image, mask))

        return image, mask
    
def convert_sequential_labels(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img_new = np.copy(img)
    u = np.unique(img)
    r = np.arange(len(u))
    for i,k in enumerate(u): img_new[np.where(img==k)] = r[i]
    return img_new

def save_array_as_image(array, name):
    img = Image.fromarray(array)
    img.save(name)
    return