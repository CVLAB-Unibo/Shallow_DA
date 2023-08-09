import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
        return image, label

class RandomCrop(object):
    def __init__(self, input_size, label_fill):
        self.size = input_size
        self.fill = 0
        self.label_fill = label_fill
        self.padding_mode = "constant"
    
    @staticmethod
    def get_params(image, output_size):
        h, w = image.size[1], image.size[0]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, label):
        #pad height
        # if (image.size[1] < self.size[0]:
        #     image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
        #     label = F.pad(label, (0, self.size[0] - label.size[1]), self.label_fill, self.padding_mode)

        # # pad width
        # if image.size[0] < self.size[1]:
        #     image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
        #     label = F.pad(label, (self.size[1] - label.size[0], 0), self.label_fill, self.padding_mode)
        if (image.size[1] < self.size[0] or image.size[0] < self.size[1]):
            image = F.resize(image, self.size, Image.BICUBIC)
            label = F.resize(label, self.size, interpolation=Image.NEAREST)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        return image, label

class Resize(object):
    def __init__(self, input_size, label_size, interpolation):
        self.input_size = input_size
        self.label_size = label_size
        self.interpolation = interpolation

    def __call__(self, image, label):
        image = F.resize(image, self.input_size, Image.BICUBIC)
        label = F.resize(label, self.label_size, interpolation=Image.NEAREST)
        return image, label

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, label):
        image = self.color_jitter(image)
        return image, label

class ToTensor(object):
    def __call__(self, image, label):
        image = F.to_tensor(image)
        label = torch.as_tensor(np.array(label), dtype=torch.long)
        return image, label

class RandomScale(object):
    def __init__(self, scale, interpolation):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, image, label):
        h, w = image.size[1], image.size[0]

        random_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        
        size = (int(h*random_scale), int(w*random_scale))
        image = F.resize(image, size, self.interpolation)
        label = F.resize(label, size, interpolation=Image.NEAREST)
        return image, label

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, label


def get_transforms(cfg):

    transform_train = [
            ToTensor(),
            Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
        ]

    if cfg.augmentations.random_flip.active:
        transform_train = [RandomHorizontalFlip(p=cfg.augmentations.random_flip.p)] + transform_train
    
    if cfg.augmentations.colorjitter.active:
        transform_train = [
                    ColorJitter(
                        brightness=cfg.augmentations.colorjitter.brightness,
                        contrast=cfg.augmentations.colorjitter.contrast,
                        saturation=cfg.augmentations.colorjitter.saturation,
                        hue=cfg.augmentations.colorjitter.hue,
                    ),
                ] + transform_train
    
    if cfg.augmentations.random_resize_crop.active:
        transform_train = [
                    Resize(input_size=tuple(cfg.dataset.input_size_train), label_size=tuple(cfg.dataset.input_size_train),
                                    interpolation=cfg.augmentations.interpolation),
                    RandomScale(scale=cfg.augmentations.random_resize_crop.scale, interpolation=cfg.augmentations.interpolation),
                    RandomCrop(input_size=tuple(cfg.dataset.input_size), label_fill=cfg.dataset.ignore_index),
                ] + transform_train
    else:
        transform_train = [Resize(input_size=tuple(cfg.dataset.input_size), label_size=tuple(cfg.dataset.out_train_size),
                                    interpolation=cfg.augmentations.interpolation)] + transform_train
    
    transform_train = Compose(transform_train)

    transform_val = [
            Resize(input_size=tuple(cfg.dataset.input_size_val), label_size=tuple(cfg.dataset.out_val_size),
                    interpolation=cfg.augmentations.interpolation),
            ToTensor(),
            Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
        ]
    transform_val = Compose(transform_val)

    transform_val_target = [
            Resize(input_size=tuple(cfg.dataset.input_size_val_target), label_size=tuple(cfg.dataset.out_val_size_target),
                    interpolation=cfg.augmentations.interpolation),
            ToTensor(),
            Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
        ]
    transform_val_target = Compose(transform_val_target)


    return transform_train, transform_val, transform_val_target