import torchvision.transforms.v2 as transforms

class GeometricTransform:
    def __init__(self, output_size=(224, 224), scale=(0.3, 2.0), ratio=(0.75, 1.33), flip=True, rotate=True, color_jitter=True, random_erasing=True):
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio
        self.flip = flip
        self.rotate = rotate
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

    def __call__(self, image, mask):
        transform = self.geometric_transform(
            output_size=self.output_size,
            scale=self.scale,
            ratio=self.ratio,
            flip=self.flip,
            rotate=self.rotate,
            color_jitter=self.color_jitter,
            random_erasing=self.random_erasing
        )
        image, mask = transform(image, mask)
        return image, mask
    
    def geometric_transform(self, output_size=(224, 224), scale=(0.3, 1.0), ratio=(0.75, 1.33), flip=True, rotate=True, color_jitter=True, random_erasing=True):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(output_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5) if flip else None,
            transforms.RandomRotation(degrees=30) if rotate else None,
        ])
        return transform
    

class ColorTransform:
    def __init__(self, brightness=0.2, contrast=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), color_jitter=True, random_erasing=True):
        self.brightness = brightness
        self.contrast = contrast
        self.scale = scale
        self.ratio = ratio
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

    def __call__(self, image, mask):
        transform = self.color_transform(
            brightness=self.brightness,
            contrast=self.contrast,
            scale=self.scale,
            ratio=self.ratio,
            color_jitter=self.color_jitter,
            random_erasing=self.random_erasing
        )
        image, mask = transform(image, mask)
        return image, mask
    
    def color_transform(self, brightness=0.2, contrast=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), color_jitter=True, random_erasing=True):
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=brightness, contrast=contrast) if color_jitter else None,
            transforms.RandomErasing(p=0.5, scale=scale, ratio=ratio, value='random') if random_erasing else None
        ])
        return transform