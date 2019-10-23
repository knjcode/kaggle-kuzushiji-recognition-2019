#!/usr/bin/env python
# coding: utf-8

# usage:
# $ python gen_single_detector_and_single_model_results.py \
# models/val_060000.pth \
# models/01_refine_efficientnet_b4_l2softmax_gray190-0060.model \
# --mode val \
# --scale-size 200 \
# --input-size 190 \
# --normalize org \
# --output-dir val_detector_060000_tta7_01_efficientnet_b4

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
parser.add_argument('--scale-size', type=int, default=None,
                    help='scale size')
parser.add_argument('--input-size', type=int, default=None,
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

    model = KModel(args.model, args.scale_size, args.input_size, normalize=args.normalize)
    kmodel_list = [[model]]
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


