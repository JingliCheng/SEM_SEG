"""

"""

import yaml
import argparse
import time
import copy
import os

# import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import unet
import unetpp
import augmentation


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, transforms_image=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.transforms_image = transforms_image
        self.images = [img for img in os.listdir(image_dir) if img.endswith('.tif')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.tif', '_m.tif'))
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)
        if self.transforms:
            for transform in self.transforms:
                image, mask = transform(image, mask)
        if self.transforms_image:
            for transform in self.transforms_image:
                image = transform(image)
        return image, mask
    
    # def image_to_tensor(self, image):
    #     return transforms.ToTensor()(image)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc


def binary_segmentation_metrics(prediction, target, threshold=0.5):
    """
    Calculate segmentation metrics for binary classification.
    Args:
    - prediction (torch.Tensor): The logits or probabilities from the model.
    - target (torch.Tensor): The ground truth binary mask.
    - threshold (float): Threshold to convert probabilities to binary output.

    Returns:
    - metrics (dict): Dictionary containing various evaluation metrics.
    """
    # Threshold predictions to create binary output
    pred = (prediction > threshold).float()

    # True positives, false positives, true negatives, false negatives
    TP = (pred + target == 2).sum().float()
    FP = (pred - target == 1).sum().float()
    TN = (pred + target == 0).sum().float()
    FN = (target - pred == 1).sum().float()

    # Metrics
    iou = TP / (TP + FP + FN) if TP + FP + FN != 0 else torch.tensor(0)
    return iou


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_loss(outputs, target, criterion):
    loss = 0
    for output in outputs:
        loss += criterion(output, target)
    return loss


def train(epoch, data_loader, model, optimizer, criterion):
    model.train()  # Set model to training mode
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad() # Clear the gradients
        outs = model(data) # Forward data batch to the model
        out = outs[0] # Get the final layer output
        loss = calculate_loss(outs, target, criterion)
        loss.backward()
        optimizer.step() # Compute gradients and update model parameters

        batch_acc = binary_segmentation_metrics(out, target)

        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc.item(), out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'IoU {IoU.val:.4f} ({IoU.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, IoU=acc))
        
        return losses.avg, acc.avg

def validate(epoch, val_loader, model, criterion):
    torch.no_grad()
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            outs = model(data)
            out = outs[0] # Get the final layer output
            loss = calculate_loss(outs, target, criterion)

        batch_acc = binary_segmentation_metrics(out, target)


        losses.update(loss.item(), out.shape[0])
        acc.update(batch_acc.item(), out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 1 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, IoU=acc))
            if epoch % 10 == 0:
                visualize_predictions(data, target, out, epoch, idx)

    print("IoU: {:.4f}".format(acc.avg))
    print("* IoU @1: {IoU.avg:.4f}".format(IoU=acc))

    return losses.avg, acc.avg


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best = 0.0
    best_model = None
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train loop
        train_loss, train_IoU = train(epoch, train_loader, model, optimizer, criterion)

        # validation loop
        print('Validation')
        val_loss, val_IoU = validate(epoch, val_loader, model, criterion)

        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('IoU/Train', train_IoU, epoch)
        writer.add_scalar('IoU/Validation', val_IoU, epoch)

        if val_IoU > best:
            best = val_IoU
            best_model = copy.deepcopy(model)

    print('Best IoU in the Validation: {:.4f}'.format(best))
    return best_model


def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)

    mean = 0.0
    var = 0.0
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        nb_samples += batch_samples
        print(nb_samples)

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)
    return mean.item(), std.item()


def visualize_predictions(inputs, labels, predictions, epoch, idx, num_images=6):
    num_images = min(num_images, inputs.size(0))  # Adjust num_images to the actual batch size if smaller
    # Threshold predictions to binary images
    predictions = (torch.sigmoid(predictions) > 0.5).float()

    inputs = inputs.cpu().detach()
    labels = labels.cpu().detach()
    predictions = predictions.cpu().detach()
    # Squeeze the channel dimension if it's 1 (for grayscale or single-channel images)
    if inputs.size(1) == 1:
        inputs = inputs.squeeze(1)
    if labels.size(1) == 1:
        labels = labels.squeeze(1)
    if predictions.size(1) == 1:
        predictions = predictions.squeeze(1)

    fig, axs = plt.subplots(3, num_images, figsize=(15, 10))
    for i in range(num_images):
        # Adjust visualization based on image type
        if inputs.ndim == 4:  # If still 4D, permute dimensions for RGB images
            axs[0, i].imshow(inputs[i].permute(1, 2, 0))
        else:  # For 2D grayscale images
            axs[0, i].imshow(inputs[i], cmap='gray')
        axs[0, i].axis('off')

        axs[1, i].imshow(labels[i], cmap='gray')
        axs[1, i].axis('off')

        # Assuming predictions are already thresholded to binary images
        axs[2, i].imshow(predictions[i], cmap='gray')
        axs[2, i].axis('off')

    plt.savefig(f'runs/validation_images/val_epoch_{epoch}_batch_{idx}.png')
    plt.close()

def main():
    global args
    global writer
    parser = argparse.ArgumentParser(description='Unet')
    parser.add_argument('--config', default='config.yaml')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        yaml_str = f.read() #config = yaml.load(f)
    config = yaml.load(yaml_str, Loader=yaml.SafeLoader)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    # Setup writer
    current_time = time.time()
    readable_time = time.ctime(current_time)
    readable_time_and_valid_for_filename = readable_time.replace('  ','').replace(' ', '_').replace(':', '-')
    writer = SummaryWriter(f'runs/{args.model}_{readable_time_and_valid_for_filename}')

    # Estimate the mean and std of the dataset
    if args.re_calculate_mean_std:
        train_dataset = SegmentationDataset(args.path_to_images, args.path_to_masks)
        mean, std = calculate_mean_std(train_dataset)
        print(mean, std)
    else:
        mean = 0.1585
        std = 0.1535

    image_geo_aug = augmentation.GeometricTransform(output_size=(768,1024))
    image_color_aug = augmentation.ColorTransform()
    image_manipulation = [image_geo_aug, image_color_aug]
    image_norm = transforms.Compose([
        transforms.Normalize(mean=[mean], std=[std]),
    ])

    # UNetPP or SimpleUNet
    if args.model == 'SimpleUNet':
        model = unet.UNet(n_channels=1, n_classes=1)
    elif args.model == 'UNetPP':
        model = unetpp.UNetPlusPlus(n_channels=1, n_classes=1, deep_supervision=args.deep_supervision)

    if torch.cuda.is_available():
        model = model.cuda()
        print('Using GPU')

    train_dataset = SegmentationDataset(args.path_to_images, args.path_to_masks, 
                                        transforms=image_manipulation, transforms_image=[image_norm])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = SegmentationDataset(args.path_to_images_val, args.path_to_masks_val, 
                                      transforms=None, transforms_image=[image_norm])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = SegmentationDataset(args.path_to_images_test, args.path_to_masks_test, 
                                       transforms=None, transforms_image=[image_norm])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss Function and Optimizer
    criterion = nn.BCEWithLogitsLoss()  # For binary classification
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.reg
    )

    # Train the model
    best_model = train_model(
        model, 
        train_loader, val_loader,
        criterion, optimizer, 
        num_epochs=args.epochs)
    writer.close()

    # Test the best model
    test_loss, test_IoU = validate(0, test_loader, best_model, criterion)
    print('IoU in the Test: {:.4f}'.format(test_IoU))

    if args.save_best:
        torch.save(best_model.state_dict(), './checkpoints/' + args.model.lower() + '.pth')

if __name__ == '__main__':
    main()
