import torchvision.transforms.v2 as transforms

class SEMImageTransform:
    def __init__(self, output_size=(224, 224), scale=(0.3, 2.0), ratio=(0.75, 1.33), flip=True, rotate=True, color_jitter=True, random_erasing=True):
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio
        self.flip = flip
        self.rotate = rotate
        self.color_jitter = color_jitter
        self.random_erasing = random_erasing

    def __call__(self, image, mask):
        transforms = self.large_scale_jitter(
            output_size=self.output_size,
            scale=self.scale,
            ratio=self.ratio,
            flip=self.flip,
            rotate=self.rotate,
            color_jitter=self.color_jitter,
            random_erasing=self.random_erasing
        )
        image, mask = transforms(image, mask)
        return image, mask
    
    def large_scale_jitter(self, output_size=(224, 224), scale=(0.3, 2.0), ratio=(0.75, 1.33), flip=True, rotate=True, color_jitter=True, random_erasing=True):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(output_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5) if flip else None,
            transforms.RandomRotation(degrees=30) if rotate else None,
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if color_jitter else None,
            # transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random') if random_erasing else None
        ])
        return transform