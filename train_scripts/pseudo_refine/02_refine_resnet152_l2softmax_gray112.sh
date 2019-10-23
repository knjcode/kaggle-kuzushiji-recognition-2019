#!/bin/bash

./train.py \
input/denoised_train_pseudo_200015779.csv \
input/denoised_valid_200015779.csv \
--resume model/02_resnet152_val15779_l2softmax_mixup_re_gray112_pseudo-resnet152-0066.model \
--batch-size 96 \
--val-batch-size 96 \
--scale-size 124 \
--input-size 112 \
--epochs 69 \
--optimizer sgd \
--wd 1e-4 \
--base-lr 0.00001 \
--warmup-epochs 5 \
--lr-step-epochs 20,40,60,70 \
--random-resized-crop-scale 0.90,0.90 \
--random-resized-crop-ratio 1.0,1.0 \
--random-horizontal-flip 0.0 \
--random-vertical-flip 0.0 \
--random-rotate-degree 0 \
--jitter-brightness 0 \
--jitter-contrast 0 \
--jitter-saturation 0 \
--jitter-hue 0 \
--pca-noise 0 \
--random-grayscale-prob 0 \
--workers 8 \
--grayscale \
--l2softmax \
--drop-last \
--onehot \
--refine \
--prefix 02_refine_resnet152_l2softmax_gray112
