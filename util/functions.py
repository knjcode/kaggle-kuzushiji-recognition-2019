#!/usr/bin/env python
# coding: utf-8

import datetime
import numbers
import os
import shutil
import sys
import warnings
import math
import pickle

import scipy.stats as stats
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from PIL import Image
from cnn_finetune import make_model

from multiprocessing import cpu_count
from torchvision import transforms
from torchvision.transforms.functional import center_crop, hflip, vflip, resize
from .metrics import L2Softmax, AdaCos, ArcFace, SphereFace, CosFace


class IncrementalVariance(object):
    def __init__(self, avg=0, count=0, var=0):
        self.avg = avg
        self.count = count
        self.var = var

    def update(self, avg, count, var):
        delta = self.avg - avg
        m_a = self.var * (self.count - 1)
        m_b = var * (count - 1)
        M2 = m_a + m_b + delta ** 2 * self.count * count / (self.count + count)
        self.var = M2 / (self.count + count - 1)
        self.avg = (self.avg * self.count + avg * count) / (self.count + count)
        self.count = self.count + count

    @property
    def average(self):
        return self.avg

    @property
    def variance(self):
        return self.var

    @property
    def std(self):
        return np.sqrt(self.var)


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.item()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def accuracy(output, target):
    pred = output.max(1, keepdim=True)[1]
    if target.ndimension() > 1:
        # onehot target
        _, tmp = target.max(1)
        return pred.eq(tmp.view_as(pred)).cpu().float().mean()
    else:
        return pred.eq(target.view_as(pred)).cpu().float().mean()


def print_batch(batch, epoch, current_num, total_num, ratio, speed, average_acc, average_loss, logger):
    logger.info('Epoch[{}] Batch[{}] [{}/{} ({:.0f}%)]\tspeed: {:.2f} samples/sec\taccuracy: {:.10f}\tloss: {:.10f}'.format(
        epoch, batch, current_num, total_num, ratio, speed, average_acc, average_loss))


def report(epoch, phase, loss_name, loss_avg, acc_name, acc_avg, logger, log_writer):
    logger.info("Epoch[{}] {}-accuracy: {}".format(epoch, phase, acc_avg))
    logger.info("Epoch[{}] {}-loss: {}".format(epoch, phase, loss_avg))
    if log_writer:
        log_writer.add_scalar(loss_name, loss_avg, epoch)
        log_writer.add_scalar(acc_name, acc_avg, epoch)


def report_lr(epoch, lr_name, lr, logger, log_writer):
    logger.info("Epoch[{}] learning-rate: {}".format(epoch, lr))
    if log_writer:
        log_writer.add_scalar(lr_name, lr, epoch)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_model(args, base_model, base_metric_fc, criterion, optimizer, scheduler, is_best, num_classes, class_names, epoch, acc1, logger):
    filepath = '{}-{}-{:04}.model'.format(args.prefix, args.model, epoch+1)
    savepath = os.path.join(args.model_dir, filepath)

    if type(base_model) == torch.nn.DataParallel:
        model_state_dict = base_model.module.state_dict()
    else:
        model_state_dict = base_model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()

    if base_metric_fc:
        if type(base_metric_fc) == torch.nn.DataParallel:
            metric_fc_state_dict = base_metric_fc.module.state_dict()
        else:
            metric_fc_state_dict = base_metric_fc.state_dict()
        for key in metric_fc_state_dict.keys():
            metric_fc_state_dict[key] = metric_fc_state_dict[key].cpu()

    state = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'arch': args.model,
        'num_classes': num_classes,
        'class_names': class_names,
        'args': args,
        'epoch': epoch + 1,
        'acc1': float(acc1)
    }

    if args.arcface or args.cosface or args.adacos or args.l2softmax:
        if args.model.startswith('efficientnet'):
            if args.l2softmax:
                state['metrics_type'] = 'embedded_l2softmax'
        else:
            state['metric_fc'] = metric_fc_state_dict
            if args.adacos or args.l2softmax:
                try:
                    state['metric_in_features'] = base_metric_fc.num_features
                except AttributeError:
                    state['metric_in_features'] = base_metric_fc.in_features
            else:
                state['metric_in_features'] = base_metric_fc.in_features
            if args.arcface:
                state['metrics_type'] = 'arcface'
            elif args.cosface:
                state['metrics_type'] = 'cosface'
            elif args.adacos:
                state['metrics_type'] = 'adacos'
            elif args.l2softmax:
                state['metrics_type'] = 'l2softmax'

    os.makedirs(args.model_dir, exist_ok=True)

    if not (args.save_best_only or args.save_best_and_last):
        torch.save(state, savepath)
        logger.info("=> Saved checkpoint to \"{}\"".format(savepath))

    if is_best:
        filepath = '{}-{}-best.model'.format(args.prefix, args.model)
        bestpath = os.path.join(args.model_dir, filepath)
        if args.save_best_only or args.save_best_and_last:
            torch.save(state, bestpath)
        else:
            shutil.copyfile(savepath, bestpath)
        logger.info("=> Saved checkpoint to \"{}\"".format(bestpath))

    if (args.epochs - 1 == epoch) and args.save_best_and_last:
        torch.save(state, savepath)
        logger.info("=> Saved checkpoint to \"{}\"".format(savepath))


def load_checkpoint(args, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=> loading saved checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint


def load_model_from_checkpoint(args, checkpoint, test_num_classes, test_class_names, grayscale=False):
    model_args = checkpoint['args']
    device = torch.device("cuda" if args.cuda else "cpu")
    model_arch = checkpoint['arch']
    num_classes = checkpoint.get('num_classes', 0)
    if num_classes == 0:
        num_classes = test_num_classes

    try:
        if model_args.grayscale:
            grayscale = True
    except AttributeError:
        grayscale = False

    if checkpoint.get('metric_fc'):
        base_model = make_model(model_arch, num_classes=num_classes, pretrained=False)
        if grayscale:
            base_model = convert_model_grayscale(model_args, model_arch, base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])
        base_model.load_state_dict(checkpoint['model'], strict=True)
        # base_metric_fc
        in_features = checkpoint['metric_in_features']
        metrics_type = checkpoint.get('metrics_type')
        if metrics_type == 'arcface':
            base_metric_fc = arcface_classifier(in_features, num_classes)
        elif metrics_type == 'cosface':
            base_metric_fc = cosface_classifier(in_features, num_classes)
        elif metrics_type == 'adacos':
            base_metric_fc = adacos_classifier(in_features, num_classes)
        elif metrics_type == 'l2softmax':
            base_metric_fc = l2softmax_classifier(in_features, num_classes)
        else:
            base_metric_fc = adacos_classifier(in_features, num_classes)
        base_metric_fc.load_state_dict(checkpoint['metric_fc'], strict=True)
    else:
        # base_model
        if model_arch.startswith('efficientnet'):
            metrics_type = checkpoint.get('metrics_type')
            if metrics_type == 'embedded_l2softmax':
                from efficientnet_l2softmax import EfficientNetL2Softmax
                base_model = EfficientNetL2Softmax.from_name(model_arch, override_params={'num_classes': num_classes})
            else:
                raise NotImplementedError('original efficientnet is not implemented yet.')
        else:
            base_model = make_model(model_arch, num_classes=num_classes, pretrained=False)
            if grayscale:
                base_model = convert_model_grayscale(args, model_arch, base_model)
        base_model.load_state_dict(checkpoint['model'], strict=True)
        base_metric_fc = None

    class_names = checkpoint.get('class_names', [])
    if len(class_names) == 0:
        class_names = test_class_names

    if args.cuda:
        model = nn.DataParallel(base_model)
        if checkpoint.get('metric_fc'):
            metric_fc = nn.DataParallel(base_metric_fc)
        else:
            metric_fc = None
    else:
        model = base_model
        if checkpoint.get('metric_fc'):
            metric_fc = base_metric_fc
        else:
            metric_fc = None

    model.to(device)
    if checkpoint.get('metric_fc'):
        metric_fc.to(device)

    if checkpoint.get('criterion'):
        criterion_state_dict = checkpoint['criterion']
    else:
        criterion_state_dict = None


    return model, metric_fc, criterion_state_dict, num_classes, class_names


def arcface_classifier(in_features, out_features, s=30.0, m=0.50, easy_margin=False):
    return ArcFace(in_features, out_features, s=s, m=m)

def cosface_classifier(in_features, out_features, s=30.0, m=0.40):
    # return AddMarginProduct(in_features, out_features, s=s, m=m)
    return CosFace(in_features, out_features, s=s, m=m)

def adacos_classifier(in_features, out_features, m=0.50):
    return AdaCos(in_features, out_features, m=m)

def l2softmax_classifier(in_features, out_features, temp=0.05):
    return L2Softmax(in_features, out_features, temp=temp)

def linear_classifier(in_features, out_features):
    return nn.Linear(in_features, out_features)

def convert_model_grayscale(args, model_arch, model):
    model_type1 = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'resnext101_32x4d', 'resnext101_64x4d', 'resnext50_32x4d',
                   'resnext101_32x8d', 'resnext101_64x4d']
    model_type2 = ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']
    model_type3 = ['inception_v3', 'inception_v4']

    if model_arch in model_type1:
        model._features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_arch in model_type2:
        model._features[0][0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_arch in model_type3:
        model._features[0].conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model._features[0].__class__ == torch.nn.modules.conv.Conv2d:
        model._features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model._features[0][0].__class__ == torch.nn.modules.conv.Conv2d:
        model._features[0][0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model


def check_args(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.mixup and args.ricap:
        warnings.warn('You can only one of the --mixup and --ricap can be activated.')
        sys.exit(1)

    if args.cutout and args.random_erasing:
        warnings.warn('You can only one of the --cutout and --random-erasing can be activated.')
        sys.exit(1)

    try:
        args.lr_step_epochs = [int(epoch) for epoch in args.lr_step_epochs.split(',')]
    except ValueError:
        warnings.warn('invalid --lr-step-epochs')
        sys.exit(1)

    try:
        args.random_resized_crop_scale = [float(scale) for scale in args.random_resized_crop_scale.split(',')]
        if len(args.random_resized_crop_scale) != 2:
            raise ValueError
    except ValueError:
        warnings.warn('invalid --random-resized-crop-scale')
        sys.exit(1)

    try:
        args.random_resized_crop_ratio = [float(ratio) for ratio in args.random_resized_crop_ratio.split(',')]
        if len(args.random_resized_crop_ratio) != 2:
            raise ValueError
    except ValueError:
        warnings.warn('invalid --random-resized-crop-ratio')
        sys.exit(1)

    if args.prefix == 'auto':
        args.prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    if args.workers is None:
        args.workers = max(1, int(0.8 * cpu_count()))
    elif args.workers == -1:
        args.workers = cpu_count()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.rgb_mean = [float(mean) for mean in args.rgb_mean.split(',')]
    args.rgb_std = [float(std) for std in args.rgb_std.split(',')]

    if args.model == 'pnasnet5large':
        scale_size = 352
        input_size = 331
    elif 'inception' in args.model:
        scale_size = 320
        input_size = 299
    elif 'xception' in args.model:
        scale_size = 320
        input_size = 299
    elif args.model.startswith('efficientnet-b1'):
        scale_size = 272
        input_size = 240
    elif args.model.startswith('efficientnet-b2'):
        scale_size = 292
        input_size = 260
    elif args.model.startswith('efficientnet-b3'):
        scale_size = 332
        input_size = 300
    elif args.model.startswith('efficientnet-b4'):
        scale_size = 412
        input_size = 380
    elif args.model.startswith('efficientnet-b5'):
        scale_size = 488
        input_size = 456
    elif args.model.startswith('efficientnet-b6'):
        scale_size = 560
        input_size = 528
    elif args.model.startswith('efficientnet-b7'):
        scale_size = 632
        input_size = 600
    else:
        scale_size = 256
        input_size = 224

    if args.scale_size:
        scale_size = args.scale_size
    else:
        args.scale_size = scale_size
    if args.input_size:
        input_size = args.input_size
    else:
        args.input_size = input_size

    if not args.cutout:
        args.cutout_holes = None
        args.cutout_length = None

    if not args.random_erasing:
        args.random_erasing_p = None
        args.random_erasing_r1 = None
        args.random_erasing_r2 = None
        args.random_erasing_sh = None
        args.random_erasing_sl = None

    if not args.mixup:
        args.mixup_alpha = None

    if not args.ricap:
        args.ricap_beta = None
        args.ricap_with_line = False

    return args


def custom_six_crop(img, size):
    """Crop the given PIL Image into custom six crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center, full)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    full = resize(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center, full)


def custom_seven_crop(img, size):
    """Crop the given PIL Image into custom seven crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center, semi_full, full)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    shift_w = int(round(w - crop_w) / 4.)
    shift_h = int(round(h - crop_h) / 4.)

    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    semi_full = resize(img.crop((shift_w, shift_h, w - shift_w, h - shift_h)), (crop_h, crop_w), interpolation=3)
    full = resize(img, (crop_h, crop_w), interpolation=3)
    return (tl, tr, bl, br, center, semi_full, full)


def custom_ten_crop(img, size):
    """Crop the given PIL Image into custom ten crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl2, tr2, bl2, br2, full)
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    shift_w = int(round(w - crop_w) / 4.)
    shift_h = int(round(h - crop_h) / 4.)

    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    tl2 = img.crop((shift_w, shift_h, crop_w + shift_w, crop_h + shift_h))  # + +
    tr2 = img.crop((w - crop_w - shift_w, shift_h, w - shift_w, crop_h + shift_h))  # - +
    bl2 = img.crop((shift_w, h - crop_h - shift_h, crop_w + shift_w, h - shift_h))  # + -
    br2 = img.crop((w - crop_w - shift_w, h - crop_h - shift_h, w - shift_w, h - shift_h))  # - -
    full = resize(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center, tl2, tr2, bl2, br2, full)


def custom_twenty_crop(img, size, vertical_flip=False):
    r"""Crop the given PIL Image into custom twenty crops.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal
    Returns:
       tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_ten = custom_ten_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_ten = custom_ten_crop(img, size)
    return first_ten + second_ten


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_labels(image_id, train_df):
    label = train_df[train_df.image_id == image_id].values[0][1]
    labels = np.array(label.split(' ')).reshape(-1, 5)
    return labels

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def score_page(preds, truth):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    tp = 0
    fp = 0
    fn = 0

    truth_indices = {
        'label': 0,
        'X': 1,
        'Y': 2,
        'Width': 3,
        'Height': 4
    }
    preds_indices = {
        'label': 0,
        'X': 1,
        'Y': 2
    }

    if pd.isna(truth) and pd.isna(preds):
        return np.array([]), {'tp': tp, 'fp': fp, 'fn': fn}

    if pd.isna(truth):
        fp += len(preds.split(' ')) // len(preds_indices)
        return np.array([]), {'tp': tp, 'fp': fp, 'fn': fn}

    if pd.isna(preds):
        fn += len(truth.split(' ')) // len(truth_indices)
        return np.array([]), {'tp': tp, 'fp': fp, 'fn': fn}

    truth = truth.split(' ')
    if len(truth) % len(truth_indices) != 0:
        raise ValueError('Malformed solution string')
    truth_label = np.array(truth[truth_indices['label']::len(truth_indices)])
    truth_xmin = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
    truth_ymin = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
    truth_ymax = truth_ymin + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

    preds = preds.split(' ')
    if len(preds) % len(preds_indices) != 0:
        raise ValueError('Malformed prediction string')
    preds_label = np.array(preds[preds_indices['label']::len(preds_indices)])
    preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
    preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)
    preds_unused = np.ones(len(preds_label)).astype(bool)

    ok_array = []
    for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
        # Matching = point inside box & character same & prediction not already used
        matching = (xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & (preds_label == label) & preds_unused
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            preds_unused[np.argmax(matching)] = False

    fp += preds_unused.sum()

    return preds_unused, {'tp': tp, 'fp': fp, 'fn': fn}

def get_center_point(bbox):
    xmin, ymin, xmax, ymax = bbox
    x_center = round((xmin + xmax) / 2)
    y_center = round((ymin + ymax) / 2)

    return (x_center, y_center)

def print_result_score(result_score, image_id, p=True):
    tp = result_score['tp']
    fp = result_score['fp']
    fn = result_score['fn']
    if (tp + fp) == 0 or (tp + fn) == 0:
        f1 = 0
        return f1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision > 0 and recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    if p:
        print('F1 score of {}: {}'.format(image_id, f1))
    return f1

def broken_box_check(boxlist, boxlist_size):
    width, height = boxlist_size
    broken_box_set = set()
    for i in range(len(boxlist)):
        current_box = boxlist[i]
        current_box = [elem.item() for elem in current_box]
        try:
            current_rect = Rectangle(*current_box)
        except ValueError:
            # boxが壊れてる場合
            broken_box_set.add(i)

    return broken_box_set

def has_intersect(a, b):
    return (max(a.x1, b.x1) <= min(a.x2, b.x2)) and (max(a.y1, b.y1) <= min(a.y2, b.y2))

def intersect(a, b):
    return Rectangle(max(a.x1, b.x1), max(a.y1, b.y1),
        min(a.x2, b.x2), min(a.y2, b.y2))

def is_rectangle(x1, y1, x2, y2):
    return x1 <= x2 and y1 <= y2

def intersect_bound(a, b):
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return x1, y1, x2, y2

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def ensemble_boxes(boxes, scores, iouThresh):
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    skip = []

    for i in range(len(boxes)-1):
        if i in skip:
            continue
        for j in range(i+1, len(boxes)):
            if j in skip:
                continue
            iou = bb_iou(boxes[i], boxes[j])
            if iou > iouThresh:
                if scores[i] >= scores[j]:
                    skip.append(j)
                else:
                    skip.append(i)

    results = set(range(len(boxes)))
    results = results - set(skip)

    target_idx = np.array(list(results))
    # print(target_idx)
    new_boxes = boxes[target_idx].astype('float')
    new_scores = scores[target_idx].astype('float')

    return new_boxes, new_scores


def ensemble_boxes_idx(boxes, scores, iouThresh, removeAloneBox=False):
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    skip = []

    for i in range(len(boxes)-1):
        max_iou = 0
        if i in skip:
            continue
        for j in range(i+1, len(boxes)):
            if j in skip:
                continue
            iou = bb_iou(boxes[i], boxes[j])
            if iou >= max_iou:
                max_iou = iou
            if iou > iouThresh:
                if scores[i] >= scores[j]:
                    skip.append(j)
                else:
                    skip.append(i)
        if removeAloneBox and max_iou == 0:
            skip.append(i)

    results = set(range(len(boxes)))
    results = results - set(skip)

    target_idx = np.array(list(results))
    # print(target_idx)
    new_boxes = boxes[target_idx].astype('float')
    new_scores = scores[target_idx].astype('float')

    if len(new_boxes) == 0:
        target_idx = np.array([])

    return new_boxes, new_scores, target_idx


def l2_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_radian(point1, point2):
    return math.atan2(point2[1] - point1[1], point2[0] - point1[0])

def get_nearest_box(boxlist, num=5):
    box_num = len(boxlist)
    nearest_dict_list = [[]] * box_num

    for i, current_box in enumerate(boxlist):
        center_point = get_center_point(current_box)
        for j, target_box in enumerate(boxlist):
            if i == j:
                continue
            target_point = get_center_point(target_box)
            cur_distance = l2_distance(center_point, target_point)
            cur_radian = get_radian(center_point, target_point)
            cur_dict = {
                'distance': cur_distance,
                'index': j,
                'radian': cur_radian
            }
            nearest_dict_list[i].extend([cur_dict])
            nearest_dict_list[i] = sorted(nearest_dict_list[i], key = lambda i: i['distance'])[0:num]

    return nearest_dict_list


def predict_image(val_images, target_index, save_dir, mode, kmodel_list, predictions, padding_rate, ensemble_iou, expand_crop, batch_size, tta, skip=True):
    image_id = val_images[target_index]
    print('current_image_id:', image_id)

    if mode == 'val':
        denoised_target_file = f'input/denoised_train/{image_id}.png'
    else:
        denoised_target_file = f'input/denoised_test/{image_id}.png'

    save_file = os.path.join(save_dir, image_id + '.pickle')
    if os.path.exists(save_file) and skip:
        return 'skip: target_file exits'

    boxlist_list = []
    bbox_score_list = []
    boxlist_size = predictions[0][target_index].size
    for prediction in predictions:
        boxlist_list.append(prediction[target_index].bbox.numpy())
        bbox_score_list.append(prediction[target_index].get_field('scores').numpy())
        if prediction[target_index].size != boxlist_size:
            raise

    cat_boxlist = np.concatenate(boxlist_list)
    cat_bbox_score = np.concatenate(bbox_score_list)

    boxlist, bbox_score = ensemble_boxes(cat_boxlist, cat_bbox_score, ensemble_iou)

    orgimg = Image.open(denoised_target_file).convert('RGB')
    orgsize = orgimg.size
    boxlist_size = predictions[0][target_index].size

    width_rate = orgsize[0] / boxlist_size[0]
    height_rate = orgsize[1] / boxlist_size[1]

    broken_box_set = broken_box_check(boxlist, boxlist_size)

    pil_image_array = []

    new_boxlist = []
    new_crop_boxlist = []
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxlist):
        if i in broken_box_set:
            continue

        x = round(float(xmin))
        y = round(float(ymin))
        xm = round(float(xmax))
        ym = round(float(ymax))
        w = round(float(xmax - xmin))
        h = round(float(ymax - ymin))

        # crop領域拡張
        if expand_crop:
            padding = round((w+h)/2 * padding_rate)
            xmin = max(x - padding, 0)
            ymin = max(y - padding, 0)
            xmax = min(x + w + padding, boxlist_size[0])
            ymax = min(y + h + padding, boxlist_size[1])

        new_bbox = (round(x*width_rate), round(y*height_rate), round(xm*width_rate), round(ym*height_rate))
        new_boxlist.append(new_bbox)
        new_crop_bbox = (round(xmin*width_rate), round(ymin*height_rate), round(xmax*width_rate), round(ymax*height_rate))
        new_crop_boxlist.append(new_crop_bbox)

        img_crop = orgimg.crop(new_crop_bbox)
        pil_image_array.append(img_crop)

    prob_list = []
    pred_labels = []
    total_output_list = []
    softmax = torch.nn.Softmax(dim=1)

    total_output = None
    for image_array in chunks(pil_image_array, batch_size):
        outputs = []

        for kmodels in kmodel_list:
            image_tensor_cuda = kmodels[0].preprocess_img_array(image_array, tta=tta)
            for kmodel in kmodels:
                output = kmodel.predict_img_tensor_cuda(image_tensor_cuda, tta=tta)
                outputs.append(output)

        total_output = sum(outputs)
        probs, labels = softmax(total_output).topk(1)
        probs = probs.cpu().numpy().flatten()
        preds = [kmodels[0].class_names[label] for label in labels]

        prob_list.extend(probs.tolist())
        pred_labels.extend(preds)
        total_output_list.extend(total_output.cpu())

    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, image_id + '.pickle')
    with open(save_file, 'wb') as f:
        r = {}
        r['size'] = orgsize
        r['prob_list'] = prob_list
        r['pred_labels'] = pred_labels
        r['bbox_score'] = bbox_score
        r['new_boxlist'] = new_boxlist
        r['new_crop_boxlist'] = new_crop_boxlist
        r['ensemble_mode'] = 'average_logits'
        pickle.dump(r, f, protocol=2)
    return f'write result: {save_file}'



class Rectangle(object):
    def __init__(self, x1, y1, x2, y2):
        if not is_rectangle(x1, y1, x2, y2):
            raise ValueError("Coordinates are invalid.\n" +
                             "Rectangle" + str((x1, y1, x2, y2)))
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        return ("Rectangle" + str((self.x1, self.y1, self.x2, self.y2)))

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def width(self):
        return (self.x2 - self.x1)

    def height(self):
        return (self.y2 - self.y1)

    def center_point(self):
        return ((self.x2 + self.x1)/2, (self.y2 + self.y1)/2)

    def vertical_bound(self, xmax, ymax):
        center_point = self.center_point()
        my_width = self.x2 - self.x1
        x1 = center_point[0] - my_width/4
        x2 = center_point[0] + my_width/4
        y1 = 0
        y2 = ymax
        return x1, y1, x2, y2


class KModel:
    def __init__(self, model_name, scale_size, input_size, normalize='org'):
        self.model_name = model_name
        self.rgb_mean = None
        self.rgb_std = None
        self.scale_size = scale_size
        self.input_size = input_size
        self.model, self.metric_fc, self.rgb_mean, self.rgb_std, self.class_names, self.grayscale = self._prepare_model(self.model_name, scale_size, input_size, normalize)
        self.preprocess, self.preprocess_tta, self.preprocess_tta7 = self.get_preprocess(self.rgb_mean, self.rgb_std)
        self.softmax = torch.nn.Softmax(dim=1)


    def _prepare_model(self, model_name, scale_size, input_size, normalize):
        args = type('', (), {})
        args.cuda = torch.cuda.is_available()
        args.scale_size = scale_size
        args.input_size = input_size
        checkpoint = load_checkpoint(args, model_name)

        model_args = checkpoint['args']
        model_arch = checkpoint['arch']
        num_classes = checkpoint.get('num_classes', 0)
        class_names = checkpoint.get('class_names', [])

        try:
            grayscale = model_args.grayscale
        except AttributeError:
            grayscale = False

        # base_model
        if model_arch.startswith('efficientnet'):
            metrics_type = checkpoint.get('metrics_type')
            if metrics_type == 'embedded_l2softmax':
                from efficientnet_l2softmax import EfficientNetL2Softmax
                base_model = EfficientNetL2Softmax.from_name(model_arch, override_params={'num_classes': num_classes, 'image_size': args.input_size})
                base_model.load_state_dict(checkpoint['model'])
                metric_fc = None
            else:
                raise NotImplementedError('original efficientnet is not implemented yet.')
        else:
            base_model = make_model(model_arch, num_classes=num_classes, pretrained=False)
            if grayscale:
                base_model = convert_model_grayscale(args, model_arch, base_model)
            if checkpoint.get('metric_fc'):
                base_model = nn.Sequential(*list(base_model.children())[:-1])
        base_model.load_state_dict(checkpoint['model'])

        # metric_fc
        if checkpoint.get('metric_fc'):
            num_classes = checkpoint.get('num_classes', 0)
            # base_metric_fc
            in_features = checkpoint['metric_in_features']
            metrics_type = checkpoint.get('metrics_type')
            if metrics_type == 'arcface':
                base_metric_fc = arcface_classifier(in_features, num_classes)
            elif metrics_type == 'cosface':
                base_metric_fc = cosface_classifier(in_features, num_classes)
            elif metrics_type == 'adacos':
                base_metric_fc = adacos_classifier(in_features, num_classes)
            elif metrics_type == 'l2softmax':
                base_metric_fc = l2softmax_classifier(in_features, num_classes)
            base_metric_fc.load_state_dict(checkpoint['metric_fc'])
            metric_fc = nn.DataParallel(base_metric_fc)
            metric_fc.eval()
            metric_fc.to('cuda')
        else:
            metric_fc = None

        rgb_mean = model_args.rgb_mean
        rgb_std = model_args.rgb_std

        if grayscale:
            if len(rgb_mean) != 1:
                if normalize == 'org':
                    rgb_mean = [0.5,]
                    rgb_std = [0.5,]
                elif normalize == 'auto':
                    rgb_mean = (rgb_mean[0], )
                    rgb_std = (rgb_std[0], )

        base_model.eval()
        model = nn.DataParallel(base_model)
        model.eval()
        if args.cuda:
            model.to('cuda')

        return model, metric_fc, rgb_mean, rgb_std, class_names, grayscale

    # アスペクト比無視して input_size にリサイズ
    def get_preprocess(self, rgb_mean, rgb_std):
        # if len(rgb_mean) == 1:
        #     gray_mean = rgb_mean
        #     gray_std = rgb_std
        #     print('use gray mean and std')
        # else:
        #     gray_mean = [0.5,]
        #     gray_std = [0.5,]

        gray_mean = rgb_mean
        gray_std = rgb_std

        if self.grayscale:
            preprocess = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self.input_size, self.input_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(gray_mean, gray_std)
            ])

            preprocess_tta = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self.scale_size, self.scale_size), interpolation=3),
                transforms.FiveCrop(self.input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ])

            preprocess_tta7 = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((self.scale_size, self.scale_size), interpolation=3),
                CustomSevenCrop(self.input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(gray_mean, gray_std)(crop) for crop in crops]))
            ])

        else:
            preprocess = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rgb_std)
            ])

            preprocess_tta = transforms.Compose([
                transforms.Resize((self.scale_size, self.scale_size), interpolation=3),
                transforms.FiveCrop(self.input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ])

            preprocess_tta7 = transforms.Compose([
                transforms.Resize((self.scale_size, self.scale_size), interpolation=3),
                CustomSevenCrop(self.input_size),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(rgb_mean, rgb_std)(crop) for crop in crops]))
            ])

        return preprocess, preprocess_tta, preprocess_tta7

    def predict(self, image_path):
        img = default_loader(image_path)
        img_tensor = self.preprocess(img).unsqueeze_(0)
        with torch.no_grad():
            output = self.model(img_tensor.cuda())
        _, labels = self.softmax(output).topk(1)
        print(labels)
        print(self.class_names[labels[0]])


    def preprocess_img_array(self, pil_img_array, tta=False):
        if tta:
            if tta == 5:
                img_tensor_list = [self.preprocess_tta(img) for img in pil_img_array]
                img_tensor_cuda = torch.stack(img_tensor_list).cuda()
            elif tta == 7:
                img_tensor_list = [self.preprocess_tta7(img) for img in pil_img_array]
                img_tensor_cuda = torch.stack(img_tensor_list).cuda()
            else:
                raise
        else:
            img_tensor_list = [self.preprocess(img) for img in pil_img_array]
            img_tensor_cuda = torch.stack(img_tensor_list).cuda()
        return img_tensor_cuda


    def predict_img_tensor_cuda(self, img_tensor_cuda, tta=False):
        with torch.no_grad():
            if tta:
                if self.grayscale:
                    if tta == 5:
                        bs, ncrops, c, h, w = len(img_tensor_cuda), 5, 1, self.input_size, self.input_size
                    elif tta == 7:
                        bs, ncrops, c, h, w = len(img_tensor_cuda), 7, 1, self.input_size, self.input_size
                    else:
                        raise
                else:
                    if tta == 5:
                        bs, ncrops, c, h, w = len(img_tensor_cuda), 5, 3, self.input_size, self.input_size
                    elif tta == 7:
                        bs, ncrops, c, h, w = len(img_tensor_cuda), 7, 3, self.input_size, self.input_size
                    else:
                        raise

                if self.metric_fc:
                    feature = self.model(img_tensor_cuda.view(-1, c, h, w))
                    output = self.metric_fc(feature.reshape(feature.shape[:-2]))
                else:
                    output = self.model(img_tensor_cuda.view(-1, c, h, w))

                output = output.view(bs, ncrops, -1).mean(1)
            else:
                if self.metric_fc:
                    feature = self.model(img_tensor_cuda)
                    output = self.metric_fc(feature.reshape(feature.shape[:-2]))
                else:
                    output = self.model(img_tensor_cuda)
        return output


    def predict_pil_array(self, pil_img_array, tta=False):
        if tta == 5:
            img_tensor_list = [self.preprocess_tta(img) for img in pil_img_array]
        elif tta == 7:
            img_tensor_list = [self.preprocess_tta7(img) for img in pil_img_array]
        else:
            img_tensor_list = [self.preprocess(img) for img in pil_img_array]
        with torch.no_grad():
            if tta:
                if tta == 5:
                    bs, ncrops, c, h, w = len(pil_img_array), 5, 3, self.input_size, self.input_size
                elif tta == 7:
                    bs, ncrops, c, h, w = len(pil_img_array), 7, 3, self.input_size, self.input_size
                else:
                    raise
                output = self.model(torch.stack(img_tensor_list).cuda().view(-1, c, h, w))
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output = self.model(torch.stack(img_tensor_list).cuda())
        return output


class CustomSixCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_six_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CustomSevenCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_seven_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CustomTenCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_ten_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CustomTwentyCrop(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return custom_twenty_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


# taken from https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

