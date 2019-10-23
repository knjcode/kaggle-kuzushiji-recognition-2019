#!/usr/bin/env python
# coding: utf-8

import math
import os

import numpy as np
import torch
import pandas as pd
import pickle
import random

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from PIL import Image


## generate train and validation image dataset
def get_image_datasets(args, scale_size, input_size, simple=False):
    interpolation = getattr(Image, args.interpolation, 3)

    if args.pca_noise > 0:
        pca_noise_prob = 1
    else:
        pca_noise_prob = 0
    lighting = Lighting(alphastd=args.pca_noise,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203]])

    # args.random_grayscale_prob = 0.5

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(args.random_rotate_degree,
                                      resample=interpolation),
            transforms.Resize((scale_size, scale_size), interpolation=interpolation),
            transforms.RandomResizedCrop(input_size,
                                         scale=args.random_resized_crop_scale,
                                         ratio=args.random_resized_crop_ratio,
                                         interpolation=interpolation),
            transforms.RandomHorizontalFlip(args.random_horizontal_flip),
            transforms.RandomVerticalFlip(args.random_vertical_flip),
            transforms.RandomGrayscale(p=args.random_grayscale_prob),
            transforms.ColorJitter(
                brightness=args.jitter_brightness,
                contrast=args.jitter_contrast,
                saturation=args.jitter_saturation,
                hue=args.jitter_hue
            ),
            transforms.ToTensor(),
            transforms.RandomApply([lighting], pca_noise_prob),
            transforms.Normalize(args.rgb_mean, args.rgb_std)
        ]),

        'valid': transforms.Compose([
            transforms.Resize((scale_size, scale_size), interpolation=interpolation),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(args.rgb_mean, args.rgb_std)
        ])
    }

    # use Cutout
    if args.cutout:
        if args.mixup or args.ricap:
            pass
            # When using mixup or ricap, cutout is applied after batch creation for learning
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.RandomRotation(args.random_rotate_degree,
                                          resample=interpolation),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.RandomResizedCrop(input_size,
                                            scale=args.random_resized_crop_scale,
                                            ratio=args.random_resized_crop_ratio,
                                            interpolation=interpolation),
                transforms.RandomHorizontalFlip(args.random_horizontal_flip),
                transforms.RandomVerticalFlip(args.random_vertical_flip),
                transforms.RandomGrayscale(p=args.random_grayscale_prob),
                transforms.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue
                ),
                transforms.ToTensor(),
                transforms.RandomApply([lighting], pca_noise_prob),
                transforms.Normalize(args.rgb_mean, args.rgb_std),
                Cutout(n_holes=args.cutout_holes, length=args.cutout_length)
            ])

    else:
        args.cutout_holes = None
        args.cutout_length = None

    # use Random erasing
    if args.random_erasing:
        if args.mixup or args.ricap:
            pass
            # When using mixup or ricap, cutout is applied after batch creation for learning
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.RandomRotation(args.random_rotate_degree,
                                          resample=interpolation),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.RandomResizedCrop(input_size,
                                            scale=args.random_resized_crop_scale,
                                            ratio=args.random_resized_crop_ratio,
                                            interpolation=interpolation),
                transforms.RandomHorizontalFlip(args.random_horizontal_flip),
                transforms.RandomVerticalFlip(args.random_vertical_flip),
                transforms.RandomGrayscale(p=args.random_grayscale_prob),
                transforms.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue
                ),
                transforms.ToTensor(),
                transforms.RandomApply([lighting], pca_noise_prob),
                transforms.Normalize(args.rgb_mean, args.rgb_std),
                RandomErasing(p=args.random_erasing_p,
                              sl=args.random_erasing_sl,
                              sh=args.random_erasing_sh,
                              r1=args.random_erasing_r1,
                              r2=args.random_erasing_r2)
            ])

    else:
        args.random_erasing_p = None
        args.random_erasing_sl = None
        args.random_erasing_sh = None
        args.random_erasing_r1 = None
        args.random_erasing_r2 = None

    if simple:
        image_datasets = {
            'train': CSVDatasetSimple(args.train, data_transforms['train']),
            'valid': CSVDatasetSimple(args.valid, data_transforms['valid'])
        }
    else:
        image_datasets = {
            'train': CSVDataset(args.train, data_transforms['train'], onehot=args.onehot, undersampling=args.undersampling),
            'valid': CSVDataset(args.valid, data_transforms['valid'], onehot=False, undersampling=False)
        }

    return image_datasets


# generate train and validation image dataset
def get_image_datasets_gray(args, scale_size, input_size):
    interpolation = getattr(Image, args.interpolation, 3)

    if args.pca_noise > 0:
        pca_noise_prob = 1
    else:
        pca_noise_prob = 0
    lighting = Lighting(alphastd=args.pca_noise,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203]])


    gray_mean = (args.rgb_mean[0], )
    gray_std = (args.rgb_std[0], )

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomRotation(args.random_rotate_degree,
                                      resample=interpolation),
            transforms.Resize((scale_size, scale_size), interpolation=interpolation),
            transforms.RandomResizedCrop(input_size,
                                         scale=args.random_resized_crop_scale,
                                         ratio=args.random_resized_crop_ratio,
                                         interpolation=interpolation),
            transforms.RandomHorizontalFlip(args.random_horizontal_flip),
            transforms.RandomVerticalFlip(args.random_vertical_flip),
            transforms.ColorJitter(
                brightness=args.jitter_brightness,
                contrast=args.jitter_contrast,
                saturation=args.jitter_saturation,
                hue=args.jitter_hue
            ),
            transforms.ToTensor(),
            transforms.Normalize(gray_mean, gray_std),
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((scale_size, scale_size), interpolation=interpolation),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(gray_mean, gray_std),
        ])
    }

    # use Cutout
    if args.cutout:
        if args.mixup or args.ricap:
            pass
            # When using mixup or ricap, cutout is applied after batch creation for learning
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomRotation(args.random_rotate_degree,
                                          resample=interpolation),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.RandomResizedCrop(input_size,
                                            scale=args.random_resized_crop_scale,
                                            ratio=args.random_resized_crop_ratio,
                                            interpolation=interpolation),
                transforms.RandomHorizontalFlip(args.random_horizontal_flip),
                transforms.RandomVerticalFlip(args.random_vertical_flip),
                transforms.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue
                ),
                transforms.ToTensor(),
                transforms.Normalize(gray_mean, gray_std),
                Cutout(n_holes=args.cutout_holes, length=args.cutout_length)
            ])

    else:
        args.cutout_holes = None
        args.cutout_length = None

    # use Random erasing
    if args.random_erasing:
        if args.mixup or args.ricap:
            pass
            # When using mixup or ricap, cutout is applied after batch creation for learning
        else:
            data_transforms['train'] = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomRotation(args.random_rotate_degree,
                                          resample=interpolation),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.RandomResizedCrop(input_size,
                                            scale=args.random_resized_crop_scale,
                                            ratio=args.random_resized_crop_ratio,
                                            interpolation=interpolation),
                transforms.RandomHorizontalFlip(args.random_horizontal_flip),
                transforms.RandomVerticalFlip(args.random_vertical_flip),
                transforms.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue
                ),
                transforms.ToTensor(),
                transforms.Normalize(gray_mean, gray_std),
                RandomErasing(p=args.random_erasing_p,
                              sl=args.random_erasing_sl,
                              sh=args.random_erasing_sh,
                              r1=args.random_erasing_r1,
                              r2=args.random_erasing_r2)
            ])

    else:
        args.random_erasing_p = None
        args.random_erasing_sl = None
        args.random_erasing_sh = None
        args.random_erasing_r1 = None
        args.random_erasing_r2 = None

    image_datasets = {
        'train': CSVDataset(args.train, data_transforms['train'], onehot=args.onehot, undersampling=args.undersampling),
        'valid': CSVDataset(args.valid, data_transforms['valid'], onehot=False, undersampling=False)
    }

    return image_datasets


# generate train and validation dataloaders
def get_dataloader(args, scale_size, input_size):
    image_datasets = get_image_datasets(args, scale_size, input_size)

    train_num_classes = len(image_datasets['train'].classes)
    val_num_classes = len(image_datasets['valid'].classes)
    assert train_num_classes == val_num_classes, 'The number of classes for train and validation is different'

    train_class_names = image_datasets['train'].classes
    val_class_names = image_datasets['valid'].classes

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=args.val_batch_size, shuffle=False, **kwargs)

    return train_loader, train_num_classes, train_class_names, \
        val_loader, val_num_classes, val_class_names


# generate train and validation dataloaders from csv input
def get_dataloader_csv(args, scale_size, input_size):
    image_datasets = get_image_datasets(args, scale_size, input_size, simple=False)

    train_num_classes = len(image_datasets['train'].classes)
    val_num_classes = len(image_datasets['valid'].classes)
    assert train_num_classes == val_num_classes, 'The number of classes for train and validation is different'

    train_class_names = image_datasets['train'].classes
    val_class_names = image_datasets['valid'].classes

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=args.val_batch_size, shuffle=False, **kwargs)

    return train_loader, train_num_classes, train_class_names, \
        val_loader, val_num_classes, val_class_names


# generate train and validation dataloaders from csv input
def get_dataloader_csv_gray(args, scale_size, input_size):
    image_datasets = get_image_datasets_gray(args, scale_size, input_size)

    train_num_classes = len(image_datasets['train'].classes)
    val_num_classes = len(image_datasets['valid'].classes)
    assert train_num_classes == val_num_classes, 'The number of classes for train and validation is different'

    train_class_names = image_datasets['train'].classes
    val_class_names = image_datasets['valid'].classes

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=args.val_batch_size, shuffle=False, **kwargs)

    return train_loader, train_num_classes, train_class_names, \
        val_loader, val_num_classes, val_class_names


# generate train and validation dataloaders from csv input
def get_dataloader_csv_simple(args, scale_size, input_size):
    image_datasets = get_image_datasets(args, scale_size, input_size, simple=True)

    train_num_classes = len(image_datasets['train'].classes)
    val_num_classes = len(image_datasets['valid'].classes)
    assert train_num_classes == val_num_classes, 'The number of classes for train and validation is different'

    train_class_names = image_datasets['train'].classes
    val_class_names = image_datasets['valid'].classes

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        image_datasets['valid'], batch_size=args.val_batch_size, shuffle=False, **kwargs)

    return train_loader, train_num_classes, train_class_names, \
        val_loader, val_num_classes, val_class_names


# https://arxiv.org/pdf/1708.04552.pdf
# modified from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask_value = img.mean()

        for n in range(self.n_holes):
            top = np.random.randint(0 - self.length // 2, h)
            left = np.random.randint(0 - self.length // 2, w)
            bottom = top + self.length
            right = left + self.length

            top = 0 if top < 0 else top
            left = 0 if left < 0 else left

            img[:, top:bottom, left:right].fill_(mask_value)

        return img


# https://arxiv.org/pdf/1708.04552.pdf
# modified from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class CutoutForBatchImages(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            batch images (Tensor): Tensor images of size (N, C, H, W).
        Returns:
            Tensor: Images with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask_value = img.mean()

        for i in range(img.size(0)):
            for n in range(self.n_holes):
                top = np.random.randint(0 - self.length // 2, h)
                left = np.random.randint(0 - self.length // 2, w)
                bottom = top + self.length
                right = left + self.length

                top = 0 if top < 0 else top
                left = 0 if left < 0 else left

                img[i, :, top:bottom, left:right].fill_(mask_value)

        return img


# https://arxiv.org/pdf/1708.04896.pdf
# modified from https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    p: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    r2: max aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with Random erasing.
        """
        if np.random.uniform(0, 1) > self.p:
            return img

        area = img.size()[1] * img.size()[2]
        for _attempt in range(100):
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(img.size()[0], h, w))
                return img

        return img


# https://arxiv.org/pdf/1708.04896.pdf
# modified from https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
class RandomErasingForBatchImages(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    p: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    r2: max aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1/0.3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        """
        Args:
            batch images (Tensor): Tensor images of size (N, C, H, W).
        Returns:
            Tensor: Images with Random erasing.
        """
        area = img.size()[2] * img.size()[3]
        for i in range(img.size(0)):

            if np.random.uniform(0, 1) > self.p:
                continue

            for _attempt in range(100):
                target_area = np.random.uniform(self.sl, self.sh) * area
                aspect_ratio = np.random.uniform(self.r1, self.r2)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[3] and h < img.size()[2]:
                    x1 = np.random.randint(0, img.size()[2] - h)
                    y1 = np.random.randint(0, img.size()[3] - w)
                    img[i, :, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, img.size()[1], h, w))
                    break

        return img


# taken from https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class CSVDataset(Dataset):
    def __init__(self, csv_file, transforms=None, onehot=False, undersampling=None):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, header=None, usecols=[0,1])
        self.df.columns = ['Unicode', 'filepath']
        self.transforms = transforms
        self.onehot = onehot
        with open('input/class_name_dict.pickle', 'rb') as f:
            self.class_name_dict = pickle.load(f)
        self.classes = list(self.class_name_dict.keys())
        if undersampling:
            self.undersample(int(undersampling))
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code = self.df['Unicode'].iloc[idx]
        label = self.class_name_dict[code]
        fname = self.df['filepath'].iloc[idx]
        img = default_loader(fname)
        processed_img = self.transforms(img)

        if self.onehot:
            onehot = torch.eye(self.num_classes)[label]
            if fname.startswith('input/pseudo_images'):
                onehot *= 0.9

            return (processed_img, onehot, fname)
        else:
            return (processed_img, label, fname)

    def undersample(self, target_count):
        # 出現回数がtarget_count回以上の文字をundersampling
        counter = self.df.Unicode.value_counts()
        code_and_count = {}
        for elem in self.df.Unicode.unique():
            if counter[elem] > target_count:
                code_and_count[elem] = counter[elem]
        for elem, count in code_and_count.items():
            drop_count = count - target_count
            target_df = self.df[self.df.Unicode == elem]
            drop_df = target_df.sample(drop_count)
            self.df = self.df.drop(drop_df.index)


class CSVDatasetSimple(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, header=None, usecols=[0,1])
        self.df.columns = ['label', 'filepath']
        self.transforms = transforms
        self.classes = [str(elem) for elem in sorted(self.df.label.unique())]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df['label'].iloc[idx]
        label_index = self.classes.index(str(label))
        fname = self.df['filepath'].iloc[idx]
        img = default_loader(fname)

        return (self.transforms(img), label_index, fname)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
