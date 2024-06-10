import torchvision.transforms.v2 as transforms
import torch
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # noise = torch.randn(tensors[0].size()) * self.std + self.mean
        # output = (tensor + noise for tensor in tensors)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class AddUniformNoise(object):
    def __init__(self, min=0.001, max=0.5):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()) * (self.max-self.min) + self.min

    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)
    

class GeometricTransform:
    def __init__(self, output_size=(224, 224), scale=(0.3, 2.0), ratio=(0.75, 1.33), flip=True, rotate=True):
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio
        self.flip = flip
        self.rotate = rotate

    def __call__(self, *args):
        transform = self.geometric_transform(
            output_size=self.output_size,
            scale=self.scale,
            ratio=self.ratio,
            flip=self.flip,
            rotate=self.rotate,
        )
        return transform(*args)
    
    def geometric_transform(self, output_size=(224, 224), scale=(0.3, 2.0), ratio=(0.75, 1.33), flip=True, rotate=True):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(output_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5) if flip else None,
            transforms.RandomRotation(degrees=30) if rotate else None,
        ])
        return transform
    

class ColorTransform:
    def __init__(self, brightness=0.2, contrast=0.2, scale=(0.01, 0.05), ratio=(0.3, 3.3), color_jitter=True, random_erasing=True):
        self.brightness = brightness
        self.contrast = contrast
        self.scale = scale
        self.ratio = ratio
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

    def __call__(self, *args):
        transform = self.color_transform(
            brightness=self.brightness,
            contrast=self.contrast,
            scale=self.scale,
            ratio=self.ratio,
            color_jitter=self.color_jitter,
            random_erasing=self.random_erasing
        )
        return transform(*args)
    
    def color_transform(self, brightness=0.2, contrast=0.2, scale=(0.01, 0.05), ratio=(0.3, 3.3), color_jitter=True, random_erasing=True):
        transform = transforms.Compose([
            # transforms.ColorJitter(brightness=brightness, contrast=contrast) if color_jitter else None,
            # AddGaussianNoise(mean=0., std=0.001),
            AddUniformNoise(min=-0.1, max=0.1),
            transforms.RandomErasing(p=0.5, scale=scale, ratio=ratio, value='random') if random_erasing else None
        ])
        return transform