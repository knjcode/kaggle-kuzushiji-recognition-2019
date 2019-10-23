#!/usr/bin/env python
# coding: utf-8

# usage:
# $ python gen_single_detector_and_multi_model_results.py \
# val_060000.pth \
# model/01_efficientnet_b4_val15779_l2softmax_mixup_re_normalize_gray190-efficientnet-b4-0061.model,\
# model/02_resnet152_val15779_l2softmax_mixup_re_gray112-resnet152-0054.model \
# --mode val \
# --scale-size 200,120 \
# --input-size 190,112 \
# --normalize org,auto \
# --output-dir val_detector_060000_tta7_first_01_efficientnet_b4

import argparse
import torch

from util.functions import Rectangle, get_labels, chunks, score_page, get_center_point, \
                           print_result_score, broken_box_check, KModel, ensemble_boxes, \
                           predict_image

parser = argparse.ArgumentParser(description='gen_single_detector_and_single_model_results')
parser.add_argument('detector', metavar='DETECTOR',
                    help='path to saved object detection results')
parser.add_argument('model', metavar='MODEL',
                    help='path to saved model')
parser.add_argument('--mode', default='val', type=str,
                    help='val or test (default: val)')
parser.add_argument('--scale-size', type=str, default=None,
                    help='scale size')
parser.add_argument('--input-size', type=str, default=None,
                    help='input size')
parser.add_argument('--normalize', default='org', type=str,
                    help='org or auto (default: org)')
parser.add_argument('--output-dir', default=None, type=str,
                    help='path to output director (default: None)')
parser.add_argument('--expand-crop', action='store_true', default=True,
                    help='expand crop area')
parser.add_argument('--padding-rate', type=float, default=0.10,
                    help='padding rate of crop area')
parser.add_argument('--tta', type=int, default=7,
                    help='use tta 7 crop (default: 7)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch-size (default: 128)')

def main():
    args = parser.parse_args()

    ensemble_iou = 0.3

    kmodel_list = []
    for model_file, scale_size, input_size, normalize in \
        zip(args.model.split(','), args.scale_size.split(','), args.input_size.split(','), args.normalize.split(',')):
        model = KModel(model_file, int(scale_size), int(input_size), normalize=normalize)
        kmodel_list.append([model])
    predictions = [torch.load(args.detector)]

    if args.mode == 'val':
        target_images = [line.rstrip() for line in open('input/val_images.list').readlines()]
    elif args.mode == 'test':
        target_images = [line.rstrip() for line in open('input/test_images.list').readlines()]
    count = len(target_images)

    for target_index in range(0, count):
        predict_image(target_images, target_index, args.output_dir, args.mode,
                      kmodel_list, predictions, args.padding_rate, ensemble_iou,
                      args.expand_crop, args.batch_size, args.tta, skip=False)


if __name__ == '__main__':
    main()


