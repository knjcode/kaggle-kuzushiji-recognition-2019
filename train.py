#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import math
import os
import sys
import time

import logzero
import numpy as np
import tensorboardX
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

from cnn_finetune import make_model
from PIL import ImageFile
from torchvision.utils import save_image
from logzero import logger
from torch.nn import functional as F

from util.dataloader import get_dataloader, get_dataloader_csv, get_dataloader_csv_simple, get_dataloader_csv_gray, CutoutForBatchImages, RandomErasingForBatchImages
from util.functions import check_args, get_lr, print_batch, report, report_lr, save_model, accuracy, Metric, arcface_classifier, cosface_classifier, adacos_classifier, l2softmax_classifier, rand_bbox, convert_model_grayscale
from util.optimizer import get_optimizer
from util.scheduler import get_cosine_annealing_lr_scheduler, get_multi_step_lr_scheduler, get_reduce_lr_on_plateau_scheduler

import signal
import warnings

signal.signal(signal.SIGINT, signal.default_int_handler)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# argparse
parser = argparse.ArgumentParser(description='train')
parser.add_argument('train', metavar='train_csv', help='path to train dataset list')
parser.add_argument('valid', metavar='valid_csv', help='path to validation dataset list')

# model architecture
parser.add_argument('--model', '-m', metavar='ARCH', default='resnet18',
                    help='specify model architecture (default: resnet18)')
parser.add_argument('--from-scratch', dest='scratch', action='store_true',
                    help='do not use pre-trained weights (default: False)')
parser.add_argument('--arcface', dest='arcface', action='store_true',
                    help='use ArcFace (Angular Margin Loss) (default: False)')
parser.add_argument('--arcface-s', type=float, default=30.0,
                    help='Feature Scale of ArcFace')
parser.add_argument('--arcface-m', type=float, default=0.50,
                    help='Margin parameter of ArcFace')
parser.add_argument('--cosface', dest='cosface', action='store_true',
                    help='use CosineFace (default: False)')
parser.add_argument('--cosface-s', type=float, default=30.0,
                    help='Feature Scale of ArcFace')
parser.add_argument('--cosface-m', type=float, default=0.40,
                    help='Margin parameter of ArcFace')
parser.add_argument('--adacos', dest='adacos', action='store_true',
                    help='use AdaCos (default: False)')
parser.add_argument('--adacos-m', type=float, default=0.50,
                    help='Margin parameter of AdaCos')
parser.add_argument('--l2softmax', dest='l2softmax', action='store_true',
                    help='use L2-constrained Softmax (default: False)')
parser.add_argument('--l2softmax-temp', type=float, default=0.05)


# epochs, batch sixe, etc
parser.add_argument('--epochs', type=int, default=30, help='number of total epochs to run (default: 30)')
parser.add_argument('--batch-size', '-b', type=int, default=128, help='the batch size (default: 128)')
parser.add_argument('--val-batch-size', type=int, default=256, help='the validation batch size (default: 256)')
parser.add_argument('-j', '--workers', type=int, default=None,
                    help='number of data loading workers (default: 80%% of the number of cores)')
parser.add_argument('--prefix', default='auto',
                    help="prefix of model and logs (default: auto)")
parser.add_argument('--log-dir', default='logs',
                    help='log directory (default: logs)')
parser.add_argument('--model-dir', default='model',
                    help='model saving dir (default: model)')
parser.add_argument('--resume', default=None, type=str, metavar='MODEL',
                    help='path to saved model (default: None)')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (default: 0)')
parser.add_argument('--disp-batches', type=int, default=0,
                    help='show progress for every n batches (default: auto)')
parser.add_argument('--save-best-only', action='store_true', default=False,
                    help='save only the latest best model according to the validation accuracy (default: False)')
parser.add_argument('--save-best-and-last', action='store_true', default=False,
                    help='save last and latest best model according to the validation accuracy (default: False)')
parser.add_argument('--drop-last', action='store_true', default=False,
                    help='drop the last incomplete batch, if the dataset size is not divisible by the batch size. (default: False)')
parser.add_argument('--grayscale', action='store_true', default=False,
                    help='change input channel from 3 to 1.')

# optimizer, lr, etc
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-factor', type=float, default=0.1,
                    help='the ratio to reduce lr on each step (default: 0.1)')
parser.add_argument('--lr-step-epochs', type=str, default='10,20',
                    help='the epochs to reduce the lr (default: 10,20)')
parser.add_argument('--lr-patience', type=int, default=None,
                    help='enable ReduceLROnPlateau lr scheduler with specified patience (default: None)')
parser.add_argument('--cosine-annealing-t-max', type=int, default=None,
                    help='enable CosineAnnealinigLR scheduler with specified T_max (default: None)')
parser.add_argument('--cosine-annealing-mult', type=int, default=2,
                    help='T_mult of CosineAnnealingLR scheduler')
parser.add_argument('--cosine-annealing-eta-min', type=float, default=1e-05,
                    help='Minimum learning rate of CosineannealingLR scheduler')
parser.add_argument('--final-lr', type=float, default=0.5,
                    help='final_lr of AdaBound optimizer')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='the optimizer type (default: sgd)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-04,
                    help='weight decay (default: 1e-04)')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs (default: 5)')

# data preprocess and augmentation settings
parser.add_argument('--scale-size', type=int, default=None,
                    help='scale size (default: auto)')
parser.add_argument('--input-size', type=int, default=None,
                    help='input size (default: auto)')
parser.add_argument('--rgb-mean', type=str, default='0,0,0',
                    help='RGB mean (default: 0,0,0)')
parser.add_argument('--rgb-std', type=str, default='1,1,1',
                    help='RGB std (default: 1,1,1)')
parser.add_argument('--random-resized-crop-scale', type=str, default='0.8,1.0',
                    help='range of size of the origin size cropped (default: 0.8,1.0)')
parser.add_argument('--random-resized-crop-ratio', type=str, default='0.75,1.3333333333333333',
                    help='range of aspect ratio of the origin aspect ratio cropped (defaullt: 0.75,1.3333333333333333)')
parser.add_argument('--random-horizontal-flip', type=float, default=0.5,
                    help='probability of the image being flipped (default: 0.5)')
parser.add_argument('--random-vertical-flip', type=float, default=0.0,
                    help='probability of the image being flipped (default: 0.0)')
parser.add_argument('--jitter-brightness', type=float, default=0.10,
                    help='jitter brightness of data augmentation (default: 0.10)')
parser.add_argument('--jitter-contrast', type=float, default=0.10,
                    help='jitter contrast of data augmentation (default: 0.10)')
parser.add_argument('--jitter-saturation', type=float, default=0.10,
                    help='jitter saturation of data augmentation (default: 0.10)')
parser.add_argument('--jitter-hue', type=float, default=0.05,
                    help='jitter hue of data augmentation (default: 0.05)')
parser.add_argument('--random-rotate-degree', type=float, default=3.0,
                    help='rotate degree of data augmentation (default: 3.0)')
parser.add_argument('--interpolation', type=str, default='BICUBIC',
                    choices=['BILINEAR', 'BICUBIC', 'NEAREST'],
                    help='interpolation. (default: BILINEAR)')
parser.add_argument('--pca-noise', type=float, default=0.,
                    help='add PCA noise (default: 0.)')
parser.add_argument('--random-grayscale-prob', type=float, default=0.,
                    help='random grayscale prob (default: 0.)')

parser.add_argument('--image-dump', action='store_true', default=False,
                    help='dump batch images and exit (default: False)')
parser.add_argument('--calc-rgb-mean-and-std', action='store_true', default=False,
                    help='calculate rgb mean and std of train images and exit (default: False)')

# misc
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. (default: None)')
parser.add_argument('--warm_restart_next', type=int, default=None,
                    help='next warm restart epoch (default: None')
parser.add_argument('--warm_restart_current', type=int, default=None,
                    help='current warm restart epoch (default: None)')
parser.add_argument('--simple-label', action='store_true', default=False,
                    help='use simple csv label (default: False)')
parser.add_argument('--onehot', action='store_true', default=False,
                    help='use onehot label (default: False)')
parser.add_argument('--freeze-bn', action='store_true',
                    help='freeze bn')
parser.add_argument('--undersampling', type=int, default=None,
                    help='undersampling')
parser.add_argument('--refine', action='store_true', default=False,
                    help='refine models pseudo images')

# cutout
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout (default: False)')
parser.add_argument('--cutout-holes', type=int, default=1,
                    help='number of holes to cut out from image (default: 1)')
parser.add_argument('--cutout-length', type=int, default=64,
                    help='length of the holes (default: 64)')

# random erasing
parser.add_argument('--random-erasing', action='store_true', default=False,
                    help='apply random erasing (default: False)')
parser.add_argument('--random-erasing-p', type=float, default=0.5,
                    help='random erasing p (default: 0.5)')
parser.add_argument('--random-erasing-sl', type=float, default=0.02,
                    help='random erasing sl (default: 0.02)')
parser.add_argument('--random-erasing-sh', type=float, default=0.4,
                    help='random erasing sh (default: 0.4)')
parser.add_argument('--random-erasing-r1', type=float, default=0.3,
                    help='random erasing r1 (default: 0.3)')
parser.add_argument('--random-erasing-r2', type=float, default=1/0.3,
                    help='random erasing r2 (default: 3.3333333333333335)')

# mixup
parser.add_argument('--mixup', action='store_true', default=False,
                    help='apply mixup (default: Falsse)')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='mixup alpha (default: 0.2)')

# ricap
parser.add_argument('--ricap', action='store_true', default=False,
                    help='apply RICAP (default: False)')
parser.add_argument('--ricap-beta', type=float, default=0.3,
                    help='RICAP beta (default: 0.3)')
parser.add_argument('--ricap-with-line', action='store_true', default=False,
                    help='RICAP with boundary line (default: False)')

# icap
parser.add_argument('--icap', action='store_true', default=False,
                    help='apply ICAP (default: False)')
parser.add_argument('--icap-beta', type=float, default=0.3,
                    help='ICAP beta (default: 0.3)')
parser.add_argument('--icap-prob', type=float, default=1.0,
                    help='ICAP probability (default: 1.0)')

# cutmix
parser.add_argument('--cutmix', action='store_true', default=False,
                    help='apply CutMix (default: False)')
parser.add_argument('--cutmix-beta', type=float, default=1.0,
                    help='CutMix beta (default: 1.0)')
parser.add_argument('--cutmix-prob', type=float, default=1.0,
                    help='CutMix probability (default: 1.0)')

best_acc1 = 0


def main():
    global args, best_acc1
    args = parser.parse_args()

    args = check_args(args)

    formatter = logging.Formatter('%(message)s')
    logzero.formatter(formatter)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    log_filename = "{}-train.log".format(args.prefix)
    logzero.logfile(os.path.join(args.log_dir, log_filename))

    # calc rgb_mean and rgb_std
    if args.calc_rgb_mean_and_std:
        calc_rgb_mean_and_std(args, logger)

    # setup dataset
    if args.simple_label:
        train_loader, train_num_classes, train_class_names, valid_loader, _valid_num_classes, _valid_class_names \
            = get_dataloader_csv_simple(args, args.scale_size, args.input_size)
    else:
        if args.grayscale:
            train_loader, train_num_classes, train_class_names, valid_loader, _valid_num_classes, _valid_class_names \
                = get_dataloader_csv_gray(args, args.scale_size, args.input_size)
        else:
            train_loader, train_num_classes, train_class_names, valid_loader, _valid_num_classes, _valid_class_names \
                = get_dataloader_csv(args, args.scale_size, args.input_size)

    if args.disp_batches == 0:
        target = len(train_loader) // 10
        args.disp_batches = target - target % 5
    if args.disp_batches < 5:
        args.disp_batches = 1

    logger.info('Running script with args: {}'.format(str(args)))
    logger.info("scale_size: {}  input_size: {}".format(args.scale_size, args.input_size))
    logger.info("rgb_mean: {}".format(args.rgb_mean))
    logger.info("rgb_std: {}".format(args.rgb_std))
    logger.info("number of train dataset: {}".format(len(train_loader.dataset)))
    logger.info("number of validation dataset: {}".format(len(valid_loader.dataset)))
    logger.info("number of classes: {}".format(len(train_class_names)))

    if args.mixup:
        logger.info("Using mixup: alpha:{}".format(args.mixup_alpha))
    if args.ricap:
        logger.info("Using RICAP: beta:{}".format(args.ricap_beta))
    if args.icap:
        logger.info("Using ICAP: prob:{} beta:{}".format(args.icap_prob, args.icap_beta))
    if args.cutmix:
        logger.info("Using CutMix: prob:{} beta:{}".format(args.cutmix_prob, args.cutmix_beta))
    if args.cutout:
        logger.info("Using cutout: holes:{} length:{}".format(args.cutout_holes, args.cutout_length))
    if args.random_erasing:
        logger.info("Using Random Erasing: p:{} s_l:{} s_h:{} r1:{} r2:{}".format(
            args.random_erasing_p, args.random_erasing_sl, args.random_erasing_sh,
            args.random_erasing_r1, args.random_erasing_r2))

    device = torch.device("cuda" if args.cuda else "cpu")

    # create  model
    if args.resume:
        # resume from a checkpoint
        if os.path.isfile(args.resume):
            logger.info("=> loading saved checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.model = checkpoint['arch']
            if args.model.startswith('efficientnet'):
                if args.l2softmax:
                    from efficientnet_l2softmax import EfficientNetL2Softmax
                    base_model = EfficientNetL2Softmax.from_name(args.model, override_params={'num_classes': train_num_classes, 'image_size': args.input_size})
                    base_metric_fc = None
                else:
                    raise NotImplementedError('original efficientnet is not implemented yet.')
            else:
                base_model = make_model(args.model,
                                        num_classes=train_num_classes,
                                        pretrained=False,
                                        input_size=(args.input_size, args.input_size))
                if args.grayscale:
                    base_model = convert_model_grayscale(args, args.model, base_model)
            if args.arcface or args.cosface or args.adacos or args.l2softmax:
                if not (args.model.startswith('efficientnet') and args.l2softmax):
                    logger.info("=> ArcFace / CosFace / AdaCos / L2Softmax")
                    base_model = nn.Sequential(*list(base_model.children())[:-1])
                    in_features = checkpoint['metric_in_features']
                    model_args = checkpoint['args']
                    if args.arcface:
                        base_metric_fc = arcface_classifier(in_features, train_num_classes,
                                                            s=model_args.arcface_s, m=model_args.arcface_m)
                    elif args.cosface:
                        base_metric_fc = cosface_classifier(in_features, train_num_classes,
                                                            s=model_args.cosface_s, m=model_args.cosface_m)
                    elif args.adacos:
                        base_metric_fc = adacos_classifier(in_features, train_num_classes, m=model_args.adacos_m)
                    elif args.l2softmax:
                        base_metric_fc = l2softmax_classifier(in_features, train_num_classes, temp=model_args.l2softmax_temp)
                    base_metric_fc.load_state_dict(checkpoint['metric_fc'])
            else:
                base_metric_fc = None
                metric_fc = None

            base_model.load_state_dict(checkpoint['model'])

            args.start_epoch = checkpoint['epoch']
            best_acc1 = float(checkpoint['acc1'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.error("=> no checkpoint found at '{}'".format(args.resume))
            sys.exit(1)
    else:
        if args.scratch and (args.arcface or args.cosface or args.adacos or args.l2softmax):
            # train from scratch with cosine-based softmax Loss
            logger.info("=> creating model '{}' with cosine-based softmax (train from scratch)".format(args.model))


            if args.model.startswith('efficientnet'):
                if args.l2softmax:
                    from efficientnet_l2softmax import EfficientNetL2Softmax
                    base_model = EfficientNetL2Softmax.from_name(args.model, override_params={'num_classes': train_num_classes, 'image_size': args.input_size})
                    base_metric_fc = None
                else:
                    raise NotImplementedError('original efficientnet is not implemented yet.')
            else:
                base_model = make_model(args.model,
                                        num_classes=train_num_classes,
                                        pretrained=False,
                                        input_size=(args.input_size, args.input_size),
                                        classifier_factory=arcface_classifier)
                if args.grayscale:
                    base_model = convert_model_grayscale(args, args.model, base_model)
                in_features = base_model._classifier.in_features
                base_model = nn.Sequential(*list(base_model.children())[:-1])
                if args.arcface:
                    base_metric_fc = arcface_classifier(in_features, train_num_classes,
                                                        s=args.arcface_s, m=args.arcface_m)
                elif args.cosface:
                    base_metric_fc = cosface_classifier(in_features, train_num_classes,
                                                        s=args.cosface_s, m=args.cosface_m)
                elif args.adacos:
                    base_metric_fc = adacos_classifier(in_features, train_num_classes,
                                                    m=args.adacos_m)
                elif args.l2softmax:
                    base_metric_fc = l2softmax_classifier(in_features, train_num_classes, temp=args.l2softmax_temp)

        elif args.scratch and (not (args.arcface or args.cosface or args.adacos or args.l2softmax)):
            # train from scratch with Softmax Loss
            logger.info("=> creating model '{}' (train from scratch)".format(args.model))
            if args.model.startswith('efficientnet'):
                raise NotImplementedError('original efficientnet is not implemented yet.')
            else:
                base_model = make_model(args.model,
                                        num_classes=train_num_classes,
                                        pretrained=False,
                                        input_size=(args.input_size, args.input_size))
                if args.grayscale:
                    base_model = convert_model_grayscale(args, args.model, base_model)
            base_metric_fc = None
            metric_fc = None
        elif (not args.scratch) and (args.arcface or args.cosface or args.adacos or args.l2softmax):

            if '.model' in args.model:
                # 既存モデルからfine-tuning
                logger.info("=> loading saved checkpoint '{}'".format(args.model))
                checkpoint = torch.load(args.model, map_location=device)
                # args.model = checkpoint['arch']
                org_num_classes = checkpoint['num_classes']
                org_model = checkpoint['arch']
                org_args = checkpoint['args']
                if org_model.startswith('efficientnet'):
                    if args.l2softmax:
                        from efficientnet_l2softmax import EfficientNetL2Softmax
                        base_model = EfficientNetL2Softmax.from_name(args.model, override_params={'num_classes': train_num_classes, 'image_size': args.input_size})
                        base_metric_fc = None
                    else:
                        raise NotImplementedError('original efficientnet is not implemented yet.')
                else:
                    base_model = make_model(org_model,
                                            num_classes=org_num_classes,
                                            pretrained=False,
                                            input_size=(org_args.input_size, org_args.input_size))
                    if args.grayscale:
                        base_model = convert_model_grayscale(args, org_model, base_model)
                    base_model = nn.Sequential(*list(base_model.children())[:-1])
                base_model.load_state_dict(checkpoint['model'])
                args.model = checkpoint['arch']

                in_features = checkpoint['metric_in_features']
                model_args = org_args

                if not org_model.startswith('efficientnet'):
                    if org_args.arcface:
                        base_metric_fc = arcface_classifier(in_features, train_num_classes,
                                                            s=model_args.arcface_s, m=model_args.arcface_m)
                    elif org_args.cosface:
                        base_metric_fc = cosface_classifier(in_features, train_num_classes,
                                                            s=model_args.cosface_s, m=model_args.cosface_m)
                    elif org_args.adacos:
                        base_metric_fc = adacos_classifier(in_features, train_num_classes, m=model_args.adacos_m)
                    elif org_args.l2softmax:
                        base_metric_fc = l2softmax_classifier(in_features, train_num_classes, temp=model_args.l2softmax_temp)
                    else:
                        base_metric_fc = None
                        metric_fc = None

                if base_metric_fc:
                    base_metric_fc.load_state_dict(checkpoint['metric_fc'])


            else:
                # fine-tuning with cosine-based softmax Loss
                logger.info("=> using pre-trained model '{}' with cosine-based softmax".format(args.model))

                if args.model.startswith('efficientnet'):
                    raise NotImplementedError('efficientnet with pre-trained is not implemented yet.')
                else:
                    base_model = make_model(args.model,
                                            num_classes=train_num_classes,
                                            pretrained=True,
                                            input_size=(args.input_size, args.input_size),
                                            classifier_factory=arcface_classifier)
                    if args.grayscale:
                        base_model = convert_model_grayscale(args, args.model, base_model)
                    in_features = base_model._classifier.in_features
                    base_model = nn.Sequential(*list(base_model.children())[:-1])
                    if args.arcface:
                        base_metric_fc = arcface_classifier(in_features, train_num_classes,
                                                            s=args.arcface_s, m=args.arcface_m)
                    elif args.cosface:
                        base_metric_fc = cosface_classifier(in_features, train_num_classes,
                                                            s=args.cosface_s, m=args.cosface_m)
                    elif args.adacos:
                        base_metric_fc = adacos_classifier(in_features, train_num_classes, m=args.adacos_m)
                    elif args.l2softmax:
                        base_metric_fc = l2softmax_classifier(in_features, train_num_classes, temp=args.l2softmax_temp)

        elif (not args.scratch) and (not (args.arcface or args.cosface or args.adacos or args.l2softmax)):

            if '.model' in args.model:
                # 既存モデルからfine-tuning
                logger.info("=> loading saved checkpoint '{}'".format(args.model))
                checkpoint = torch.load(args.model, map_location=device)
                # args.model = checkpoint['arch']
                org_num_classes = checkpoint['num_classes']
                org_model = checkpoint['arch']
                org_args = checkpoint['args']
                if org_model.startswith('efficientnet'):
                    raise NotImplementedError('original efficientnet is not implemented yet.')
                else:
                    base_model = make_model(org_model,
                                            num_classes=org_num_classes,
                                            pretrained=False,
                                            input_size=(org_args.input_size, org_args.input_size))
                    if args.grayscale:
                        base_model = convert_model_grayscale(args, org_model, base_model)
                base_model.load_state_dict(checkpoint['model'])
                if org_num_classes != train_num_classes:
                    if org_model.startswith('efficientnet'):
                        num_features = base_model._fc.in_features
                        base_model._fc = nn.Linear(num_features, train_num_classes)
                    else:
                        num_features = base_model._classifier.in_features
                        base_model._classifier = nn.Linear(num_features, train_num_classes)
                        base_model.num_classes = train_num_classes
                args.model = checkpoint['arch']
                base_metric_fc = None
            else:
                # fine-tuning with Softmax Loss
                logger.info("=> using pre-trained model '{}'".format(args.model))
                if args.model.startswith('efficientnet'):
                    raise NotImplementedError('original efficientnet is not implemented yet.')
                else:
                    base_model = make_model(args.model,
                                            num_classes=train_num_classes,
                                            pretrained=True,
                                            input_size=(args.input_size, args.input_size))
                    if args.grayscale:
                        base_model = convert_model_grayscale(args, args.model, base_model)
                base_metric_fc = None
                metric_fc = None


    if args.cuda:
        logger.info("=> using GPU")
        model = nn.DataParallel(base_model)
        model.to(device)
        if base_metric_fc:
            metric_fc = nn.DataParallel(base_metric_fc)
            metric_fc.to(device)
        else:
            metric_fc = None
    else:
        logger.info("=> using CPU")
        model = base_model
        metric_fc = base_metric_fc

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model, metric_fc)
    logger.info('=> using optimizer: {}'.format(args.optimizer))
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> restore optimizer state from checkpoint")

    # create scheduler
    if args.lr_patience:
        scheduler = get_reduce_lr_on_plateau_scheduler(args, optimizer)
        logger.info("=> using ReduceLROnPlateau scheduler")
    elif args.cosine_annealing_t_max:
        scheduler = get_cosine_annealing_lr_scheduler(args, optimizer, args.cosine_annealing_t_max, len(train_loader))
        logger.info("=> using CosineAnnealingLR scheduler")
    else:
        scheduler = get_multi_step_lr_scheduler(args, optimizer, args.lr_step_epochs, args.lr_factor)
        logger.info("=> using MultiStepLR scheduler")
    if args.resume:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("=> restore lr scheduler state from checkpoint")

    if args.refine:
        set_learning_rate(optimizer, args.base_lr)

    logger.info("=> model and logs prefix: {}".format(args.prefix))
    logger.info("=> log dir: {}".format(args.log_dir))
    logger.info("=> model dir: {}".format(args.model_dir))
    tensorboradX_log_dir = os.path.join(args.log_dir, "{}-tensorboardX".format(args.prefix))
    log_writer = tensorboardX.SummaryWriter(tensorboradX_log_dir)
    logger.info("=> tensorboardX log dir: {}".format(tensorboradX_log_dir))

    if args.cuda:
        cudnn.benchmark = True

    # for CosineAnnealingLR
    if args.resume:
        args.warm_restart_next = checkpoint['args'].warm_restart_next
        args.warm_restart_current = checkpoint['args'].warm_restart_current
    else:
        if args.cosine_annealing_t_max:  # CosineAnnealingLR
            args.warm_restart_next = args.cosine_annealing_t_max + args.warmup_epochs
            args.warm_restart_current = args.warmup_epochs

    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        # CosineAnnealingLR warm restart
        if args.cosine_annealing_t_max and (epoch % args.warm_restart_next == 0) and epoch != 0:
            current_span = args.warm_restart_next - args.warm_restart_current
            next_span = current_span * args.cosine_annealing_mult
            args.warm_restart_current = args.warm_restart_next
            args.warm_restart_next = args.warm_restart_next + next_span
            scheduler = get_cosine_annealing_lr_scheduler(args, optimizer, next_span, len(train_loader))

        if args.mixup:
            train(args, 'mixup', train_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer)
        elif args.ricap:
            train(args, 'ricap', train_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer)
        elif args.icap:
            train(args, 'icap', train_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer)
        elif args.cutmix:
            train(args, 'cutmix', train_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer)
        else:
            train(args, 'normal', train_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer)

        report_lr(epoch, 'x_learning_rate', get_lr(optimizer), logger, log_writer)

        acc1 = valid(args, valid_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer)

        elapsed_time = time.time() - start
        logger.info("Epoch[{}] Time cost: {} [sec]".format(epoch, elapsed_time))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_model(args, base_model, base_metric_fc, criterion, optimizer, scheduler, is_best, train_num_classes, train_class_names, epoch, acc1, logger)

        # reset dataset
        if args.undersampling:
            logger.info("reset dataloader")
            if args.grayscale:
                train_loader, train_num_classes, train_class_names, valid_loader, _valid_num_classes, _valid_class_names \
                    = get_dataloader_csv_gray(args, args.scale_size, args.input_size)
            else:
                train_loader, train_num_classes, train_class_names, valid_loader, _valid_num_classes, _valid_class_names \
                    = get_dataloader_csv(args, args.scale_size, args.input_size)
            logger.info("number of train dataset: {}".format(len(train_loader.dataset)))
            logger.info("number of validation dataset: {}".format(len(valid_loader.dataset)))


def train(args, train_mode, train_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer):
    total_size = 0
    data_size = len(train_loader.dataset)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    model.train()
    if metric_fc:
        metric_fc.train()

    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    start = time.time()


    if train_mode in ['mixup', 'ricap', 'icap']:
        if args.cutout:
            batch_cutout = CutoutForBatchImages(n_holes=args.cutout_holes, length=args.cutout_length)
        if args.random_erasing:
            batch_random_erasing = RandomErasingForBatchImages(p=args.random_erasing_p,
                                                            sl=args.random_erasing_sl,
                                                            sh=args.random_erasing_sh,
                                                            r1=args.random_erasing_r1,
                                                            r2=args.random_erasing_r2)

    for batch_idx, (data, target, _paths) in enumerate(train_loader):
        # adjust_learning_rate(args, epoch, batch_idx, train_loader, optimizer, scheduler, logger)

        if train_mode is 'mixup':
            alpha = args.mixup_alpha
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
            index = torch.randperm(data.size(0))
            mixed_data = lam * data + (1 - lam) * data[index, :]
            target_a, target_b = target, target[index]

            if args.cutout:
                mixed_data = batch_cutout(mixed_data)
            if args.random_erasing:
                mixed_data = batch_random_erasing(mixed_data)

        elif train_mode is 'ricap':
            beta = args.ricap_beta
            I_x, I_y = data.size()[2:]
            w = int(np.round(I_x * np.random.beta(beta, beta)))
            h = int(np.round(I_y * np.random.beta(beta, beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]
            cropped_images = {}
            c_ = {}
            W_ = {}
            if args.cuda:
                data = data.cuda(non_blocking=True)
            for k in range(4):
                index = torch.randperm(data.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = data[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                if args.cuda:
                    c_[k] = target[index].cuda(non_blocking=True)
                else:
                    c_[k] = target[index]
                W_[k] = w_[k] * h_[k] / (I_x * I_y)
            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)
            # draw lines
            if args.ricap_with_line:
                patched_images[:, :, w-1:w+1, :] = 0.
                patched_images[:, :, :, h-1:h+1] = 0.

            if args.cutout:
                patched_images = batch_cutout(patched_images)
            if args.random_erasing:
                patched_images = batch_random_erasing(patched_images)

        elif train_mode is 'icap':
            p = args.icap_prob
            beta = args.icap_beta
            r = np.random.rand(1)
            if r < p:
                I_x, I_y = data.size()[2:]
                w = int(np.round(I_x * np.random.beta(beta, beta)))
                h = int(np.round(I_y * np.random.beta(beta, beta)))
                h_from = [0, 0, h, h]
                h_to = [h, h, I_y, I_y]
                w_from = [0, w, 0, w]
                w_to = [w, I_x, w, I_x]
                cropped_images = {}
                c_ = {}
                W_ = {}

                if args.cuda:
                    data = data.cuda(non_blocking=True)
                for k in range(4):
                    index = torch.randperm(data.size(0))
                    cropped_images[k] = data[index][:, :, h_from[k]:h_to[k], w_from[k]:w_to[k]]
                    if args.cuda:
                        c_[k] = target[index].cuda(non_blocking=True)
                    else:
                        c_[k] = target[index]
                    W_[k] = (h_to[k] - h_from[k]) * (w_to[k] - w_from[k]) / (I_x * I_y)

                patched_images = torch.cat(
                    (torch.cat((cropped_images[0], cropped_images[2]), 2),
                    torch.cat((cropped_images[1], cropped_images[3]), 2)), 3)

                if args.cutout:
                    patched_images = batch_cutout(patched_images)
                if args.random_erasing:
                    patched_images = batch_random_erasing(patched_images)
            else:
                # change normal train mode
                train_mode = 'normal'

        elif train_mode is 'cutmix':
            p = args.cutmix_prob
            beta = args.cutmix_beta
            r = np.random.rand(1)
            if beta > 0 and r < p:
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                index = torch.randperm(data.size(0))
                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
                mixed_data = data
                target_a, target_b = target, target[index]
            else:
                # change normal train mode
                train_mode = 'normal'


        # normal train mode applies Cutout or Random Erasing inside dataloader

        if args.image_dump:
            if train_mode in ['mixup', 'cutmix']:
                save_image(mixed_data, './samples.jpg')
            elif train_mode in ['ricap', 'icap']:
                save_image(patched_images, './samples.jpg')
            else:
                save_image(data, './samples.jpg')
            logger.info("image saved! at ./samples.jpg")
            sys.exit(0)

        if args.cuda:
            if train_mode in ['mixup', 'cutmix']:
                mixed_data = mixed_data.cuda(non_blocking=True)
                target_a = target_a.cuda(non_blocking=True)
                target_b = target_b.cuda(non_blocking=True)
            elif train_mode in ['ricap', 'icap']:
                patched_images = patched_images.cuda(non_blocking=True)
            else:  # vanila train
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        # output and loss
        if train_mode in ['mixup', 'cutmix']:
            if args.l2softmax and (not args.model.startswith('efficientnet')):
                feature = model(mixed_data)
                output = metric_fc(feature.reshape(feature.shape[:-2]))
            else:
                output = model(mixed_data)

            if args.onehot:
                log_prob = F.log_softmax(output, dim=1)
                loss = (lam * (-(target_a * log_prob).sum(dim=1)) + (1 - lam) * (-(target_b * log_prob).sum(dim=1))).sum()
            else:
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

        elif train_mode in ['ricap', 'icap']:
            if args.arcface or args.cosface or args.adacos:
                feature = model(patched_images)
                output = metric_fc(feature.reshape(feature.shape[:-2]), target)
            elif args.l2softmax:
                feature = model(patched_images)
                output = metric_fc(feature.reshape(feature.shape[:-2]))
            else:
                output = model(patched_images)
            if args.onehot:
                log_prob = F.log_softmax(output, dim=1)
                loss = sum(W_[k] * -(c_[k] * log_prob).sum(dim=1) for k in range(4))
            else:
                loss = sum(W_[k] * criterion(output, c_[k]) for k in range(4))
        else:  # vanila train
            if args.model.startswith('efficientnet'):
                if args.arcface or args.cosface or args.adacos:
                    feature = model.module.extract_features(data)
                    feature = torch.mean(feature, (2,3))
                    output = metric_fc(feature.reshape(feature.shape[:-2]), target)
                elif args.l2softmax:
                    output = model(data)
                else:
                    output = model(data)
            else:
                if args.arcface or args.cosface or args.adacos:
                    feature = model(data)
                    output = metric_fc(feature.reshape(feature.shape[:-2]), target)
                elif args.l2softmax:
                    feature = model(data)
                    output = metric_fc(feature.reshape(feature.shape[:-2]))
                else:
                    output = model(data)

            if args.onehot:
                log_prb = F.log_softmax(output, dim=1)
                loss = -(target * log_prb).sum(dim=1)
                loss = loss.sum()
            else:
                loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        adjust_learning_rate(args, epoch, batch_idx, train_loader, optimizer, scheduler, logger)

        if train_mode in ['mixup', 'cutmix']:
            train_accuracy.update(lam * accuracy(output, target_a) + (1 - lam) * accuracy(output, target_b))
        elif train_mode in ['ricap', 'icap']:
            train_accuracy.update(sum([W_[k] * accuracy(output, c_[k]) for k in range(4)]))
        else:  # vanila train
            train_accuracy.update(accuracy(output, target))

        train_loss.update(loss)

        total_size += data.size(0)
        if (batch_idx + 1) % args.disp_batches == 0:
            prop = 100. * (batch_idx+1) / len(train_loader)
            elapsed_time = time.time() - start
            speed = total_size / elapsed_time
            print_batch(batch_idx+1, epoch, total_size, data_size, prop, speed, train_accuracy.avg, train_loss.avg, logger)

    report(epoch, 'Train', 'train/loss', train_loss.avg, 'train/accuracy', train_accuracy.avg, logger, log_writer)


def valid(args, valid_loader, model, metric_fc, device, criterion, optimizer, scheduler, epoch, logger, log_writer):
    valid_loss = Metric('valid_loss')
    valid_accuracy = Metric('valid_accuracy')
    model.eval()
    if metric_fc:
        metric_fc.eval()

    with torch.no_grad():
        for (data, target, _paths) in valid_loader:
            if args.cuda:
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            if args.model.startswith('efficientnet'):
                if args.arcface or args.cosface or args.adacos:
                    feature = model.module.extract_features(data)
                    feature = torch.mean(feature, (2,3))
                    output = metric_fc(feature.reshape(feature.shape[:-2]))
                elif args.l2softmax:
                    output = model(data)
                else:
                    output = model(data)
            else:
                if args.arcface or args.cosface or args.adacos:
                    feature = model(data)
                    output = metric_fc(feature.reshape(feature.shape[:-2]))
                elif args.l2softmax:
                    feature = model(data)
                    output = metric_fc(feature.reshape(feature.shape[:-2]))
                else:
                    output = model(data)

            loss = criterion(output, target)

            valid_accuracy.update(accuracy(output, target))
            valid_loss.update(loss)

        report(epoch, 'Validation', 'val/loss', valid_loss.avg, 'val/accuracy', valid_accuracy.avg, logger, log_writer)

        if args.lr_patience:  # ReduceLROnPlateau
            scheduler.step(valid_loss.avg)
        elif not args.cosine_annealing_t_max:  # MultiStepLR
            scheduler.step()

    return valid_accuracy.avg


def get_warmup_lr_adj(args, epoch, batch_idx, train_loader, optimizer, logger):
    lr_adj = 1.
    if epoch < args.warmup_epochs:
        epoch = epoch * len(train_loader)
        epoch += float(batch_idx + 1)
        lr_adj = epoch / (len(train_loader) * (args.warmup_epochs + 1))
    return lr_adj


def adjust_learning_rate(args, epoch, batch_idx, train_loader, optimizer, scheduler, logger):
    lr_adj = 1.
    if epoch < args.warmup_epochs:
        lr_adj = get_warmup_lr_adj(args, epoch, batch_idx, train_loader, optimizer, logger)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.base_lr * lr_adj
    else:
        if args.cosine_annealing_t_max:
            scheduler.step()


def calc_rgb_mean_and_std(args, logger):
    from util.dataloader import get_image_datasets_for_rgb_mean_and_std
    from util.functions import IncrementalVariance
    from tqdm import tqdm

    image_datasets = get_image_datasets_for_rgb_mean_and_std(args, args.scale_size, args.input_size)
    logger.info("=> Calculate rgb mean and std (dir: {}  images: {}  batch-size: {})".format(args.train, len(image_datasets), args.batch_size))

    if args.batch_size < len(image_datasets):
        logger.info("To calculate more accurate values, please specify as large a batch size as possible.")

    kwargs = {'num_workers': args.workers}
    train_loader = torch.utils.data.DataLoader(
        image_datasets, batch_size=args.batch_size, shuffle=False, **kwargs)

    iv = IncrementalVariance()
    processed = 0
    with tqdm(total=len(train_loader), desc="Calc rgb mean/std") as t:
        for data, _target in train_loader:
            numpy_images = data.numpy()
            batch_mean = np.mean(numpy_images, axis=(0, 2, 3))
            batch_var = np.var(numpy_images, axis=(0, 2, 3))
            iv.update(batch_mean, len(numpy_images), batch_var)
            processed += len(numpy_images)
            t.update(1)

    logger.info("=> processed: {} images".format(processed))
    logger.info("=> calculated rgb mean: {}".format(iv.average))
    logger.info("=> calculated rgb std: {}".format(iv.std))

    np.set_printoptions(formatter={'float': '{:0.3f}'.format})
    rgb_mean_option = np.array2string(iv.average, separator=',').replace('[', '').replace(']', '')
    rgb_std_option = np.array2string(iv.std, separator=',').replace('[', '').replace(']', '')
    logger.info("Please use following command options when train and test:")
    logger.info(" --rgb-mean {} --rgb-std {}".format(rgb_mean_option, rgb_std_option))

    sys.exit(0)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
