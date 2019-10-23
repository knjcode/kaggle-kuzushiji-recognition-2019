#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import logging
import os
import pickle
import signal
import warnings

import logzero
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from multiprocessing import cpu_count
from PIL import Image, ImageFile
from sklearn.metrics import classification_report
from torchvision import transforms
from tqdm import tqdm
from logzero import logger
from torch.nn import functional as F

from util.dataloader import ImageFolderWithPaths, CSVDataset
from util.functions import accuracy, load_checkpoint, load_model_from_checkpoint, Metric, CustomTenCrop, CustomTwentyCrop, CustomSixCrop, CustomSevenCrop

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
signal.signal(signal.SIGINT, signal.default_int_handler)
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='test')
parser.add_argument('test', metavar='valid_csv', help='path to test dataset list')
parser.add_argument('--prefix', default='auto',
                    help="prefix of model and logs (default: auto)")
parser.add_argument('--log-dir', default='logs',
                    help='log directory (default: logs)')
parser.add_argument('--model', '-m', type=str,
                    help='model file to test')
parser.add_argument('-j', '--workers', type=int, default=None,
                    help='number of data loading workers (default: 80%% of the number of cores)')

parser.add_argument('-b', '--batch-size', type=int, default=128, help='the batch size')
parser.add_argument('--topk', type=int, default=3,
                    help='report the top-k accuracy (default: 3)')
parser.add_argument('--print-cr', action='store_true', default=False,
                    help='print classification report (default: False)')
parser.add_argument('--onehot', action='store_true', default=False,
                    help='use onehot label (default: False)')


# Test Time Augmentation
parser.add_argument('--tta', action='store_true', default=False,
                    help='test time augmentation (use FiveCrop)')
parser.add_argument('--tta-ten-crop', action='store_true', default=False,
                    help='test time augmentation (use TenCrop)')
parser.add_argument('--tta-custom-six-crop', action='store_true', default=False,
                    help='test time augmentation (use CustomSixCrop)')
parser.add_argument('--tta-custom-seven-crop', action='store_true', default=False,
                    help='test time augmentation (use CustomSevenCrop)')
parser.add_argument('--tta-custom-ten-crop', action='store_true', default=False,
                    help='test time augmentation (use CustomTenCrop)')
parser.add_argument('--tta-custom-twenty-crop', action='store_true', default=False,
                    help='test time augmentation (use CustomTwentyCrop)')

# data preprocess
parser.add_argument('--scale-size', type=int, default=None,
                    help='scale size (default: auto)')
parser.add_argument('--input-size', type=int, default=None,
                    help='input size (default: auto)')
parser.add_argument('--rgb-mean', type=str, default=None,
                    help='RGB mean (default: auto)')
parser.add_argument('--rgb-std', type=str, default=None,
                    help='RGB std (default: auto)')
parser.add_argument('--interpolation', type=str, default=None,
                    choices=[None, 'BILINEAR', 'BICUBIC', 'NEAREST'],
                    help='interpolation. (default: auto)')
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='change input channel from 3 to 1.')

# misc
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.prefix == 'auto':
        args.prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    formatter = logging.Formatter('%(message)s')
    logzero.formatter(formatter)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    log_filename = "{}-test.log".format(args.prefix)
    log_filepath = os.path.join(args.log_dir, log_filename)
    logzero.logfile(log_filepath)

    if args.workers is None:
        args.workers = max(1, int(0.8 * cpu_count()))
    elif args.workers == -1:
        args.workers = cpu_count()

    cudnn.benchmark = True

    logger.info('Running script with args: {}'.format(str(args)))

    checkpoint = load_checkpoint(args, args.model)
    logger.info("=> loaded the model (epoch {})".format(checkpoint['epoch']))
    model_arch = checkpoint['arch']
    model_args = checkpoint['args']

    if model_arch.startswith('efficientnet-b4'):
        scale_size = 200
        input_size = 190
    else:
        scale_size = 120
        input_size = 112

    if args.scale_size:
        scale_size = args.scale_size
    else:
        args.scale_size = scale_size
    if args.input_size:
        input_size = args.input_size
    else:
        args.input_size = input_size

    if args.rgb_mean:
        rgb_mean = args.rgb_mean
        rgb_mean = [float(mean) for mean in rgb_mean.split(',')]
    else:
        rgb_mean = model_args.rgb_mean

    if args.rgb_std:
        rgb_std = args.rgb_std
        rgb_std = [float(std) for std in rgb_std.split(',')]
    else:
        rgb_std = model_args.rgb_std

    if args.interpolation:
        interpolation = args.interpolation
    else:
        try:
            interpolation = model_args.interpolation
        except AttributeError:
            interpolation = 'BICUBIC'

    logger.info("scale_size: {}  input_size: {}".format(scale_size, input_size))
    logger.info("rgb_mean: {}".format(rgb_mean))
    logger.info("rgb_std: {}".format(rgb_std))
    logger.info("interpolation: {}".format(interpolation))

    interpolation = getattr(Image, interpolation, 3)

    try:
        args.grayscale = model_args.grayscale
    except:
        pass

    # Data augmentation and normalization for test
    if args.grayscale:
        if len(rgb_mean) == 1:
            gray_mean = rgb_mean
            gray_std = rgb_std
        else:
            # gray_mean = [0.5,]
            # gray_std = [0.5,]
            gray_mean = (rgb_mean[0], )
            gray_std = (rgb_std[0], )

        data_transforms = {
            'test': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(gray_mean, gray_std),
            ]),
            'test_FiveCrop': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.FiveCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ]),
            'test_TenCrop': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.TenCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ]),
            'test_CustomSixCrop': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomSixCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ]),
            'test_CustomSevenCrop': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomSevenCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ]),
            'test_CustomTenCrop': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomTenCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ]),
            'test_CustomTwentyCrop': transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomTwentyCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ])
        }
    else:
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rgb_std)
            ]),
            'test_FiveCrop': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.FiveCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ]),
            'test_TenCrop': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                transforms.TenCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ]),
            'test_CustomSixCrop': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomSixCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ]),
            'test_CustomSevenCrop': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomSevenCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ]),
            'test_CustomTenCrop': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomTenCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ]),
            'test_CustomTwentyCrop': transforms.Compose([
                transforms.Resize((scale_size, scale_size), interpolation=interpolation),
                CustomTwentyCrop(input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ])
        }

    tfms = 'test'
    if args.tta:
        tfms = 'test_FiveCrop'
        batch_size = args.batch_size // 5
    elif args.tta_ten_crop:
        tfms = 'test_TenCrop'
        batch_size = args.batch_size // 10
    elif args.tta_custom_six_crop:
        tfms = 'test_CustomSixCrop'
        batch_size = args.batch_size // 6
    elif args.tta_custom_seven_crop:
        tfms = 'test_CustomSevenCrop'
        batch_size = args.batch_size // 7
    elif args.tta_custom_ten_crop:
        tfms = 'test_CustomTenCrop'
        batch_size = args.batch_size // 10
    elif args.tta_custom_twenty_crop:
        tfms = 'test_CustomTwentyCrop'
        batch_size = args.batch_size // 20
    else:
        batch_size = args.batch_size


    image_datasets = {
        'test': CSVDataset(args.test, data_transforms[tfms], onehot=args.onehot)
    }

    test_num_classes = len(image_datasets['test'].classes)
    test_class_names = image_datasets['test'].classes

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        image_datasets['test'], batch_size=batch_size, shuffle=False, **kwargs)

    logger.info("number of test dataset: {}".format(len(test_loader.dataset)))
    logger.info("number of classes: {}".format(len(test_class_names)))

    model, metric_fc, criterion_state_dict, num_classes, class_names = load_model_from_checkpoint(args, checkpoint, test_num_classes, test_class_names, grayscale=args.grayscale)

    if args.topk > num_classes:
        logger.warn('--topk must be less than or equal to the class number of the model')
        args.topk = num_classes
        logger.warn('--topk set to {}'.format(num_classes))

    # check test and train class names
    do_report = True
    if test_num_classes != num_classes:
        logger.info("The number of classes for train and test is different.")
        logger.info("Skip accuracy report.")
        do_report = False

    test(args, model_arch, model, metric_fc, test_loader, class_names, do_report, logger)

    logger.info("=> Saved test log to \"{}\"".format(log_filepath))


def test(args, model_arch, model, metric_fc, test_loader, class_names, do_report, logger):
    model.module.eval()
    if metric_fc:
        metric_fc.module.eval()
    test_accuracy = Metric('test_accuracy')
    test_loss = Metric('test_loss')


    pred = []
    Y = []
    correct_num = 0

    filepath = '{}-test-results.log'.format(args.prefix)
    savepath = os.path.join(args.log_dir, filepath)
    f = open(savepath, 'w')

    softmax = torch.nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()

    with tqdm(total=len(test_loader), desc='Test') as t:
        with torch.no_grad():
            for (data, target, paths) in test_loader:
                if args.cuda:
                    data = data.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                if args.tta or args.tta_ten_crop or \
                   args.tta_custom_ten_crop or args.tta_custom_twenty_crop or \
                   args.tta_custom_six_crop or args.tta_custom_seven_crop:
                    bs, ncrops, c, h, w = data.size()
                    if metric_fc:
                        feature = model(data.view(-1, c, h, w))
                        output = metric_fc(feature.reshape(feature.shape[:-2]))
                    else:
                        output = model(data.view(-1, c, h, w))
                    output = output.view(bs, ncrops, -1).mean(1)
                else:
                    if metric_fc:
                        feature = model(data)
                        output = metric_fc(feature.reshape(feature.shape[:-2]))
                    else:
                        output = model(data)

                if do_report:
                    pred += [int(l.argmax()) for l in output]
                    Y += [int(l) for l in target]

                for path, y, preds in zip(paths, target, softmax(output)):
                    probabilities, labels = preds.topk(args.topk)
                    preds_text = ''
                    for i in range(args.topk):
                        preds_text += " {} {}".format(labels[i], probabilities[i])
                    f.write("{} {}{}\n".format(path, int(y), preds_text))

                    if str(y.item()) == str(labels[0].item()):
                        correct_num += 1

                if do_report:
                    test_accuracy.update(accuracy(output, target))
                    test_loss.update(criterion(output, target))
                    t.set_postfix({'loss': test_loss.avg.item(),
                                  'accuracy': 100. * test_accuracy.avg.item()})
                t.update(1)

    f.close()
    logger.info("=> Saved test results to \"{}\"".format(savepath))

    if do_report:

        cr_filepath = '{}-test-classification_report.log'.format(args.prefix)
        cr_savepath = os.path.join(args.log_dir, cr_filepath)

        cr = classification_report(Y, pred, target_names=class_names)
        if args.print_cr:
            print(cr)
        with open(cr_savepath, 'w') as crf:
            crf.write(cr)
        logger.info("=> Saved classification report to \"{}\"".format(cr_savepath))

        logger.info("model: {}".format(args.model))
        logger.info("Test-loss: {}".format(test_loss.avg))
        logger.info("Test-accuracy: {} ({}/{})".format((correct_num / len(test_loader.dataset)), correct_num, len(test_loader.dataset)))



if __name__ == '__main__':
    main()
