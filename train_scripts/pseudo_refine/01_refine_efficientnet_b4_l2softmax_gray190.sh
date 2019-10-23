#!/bin/bash

./train.py \
input/denoised_train_pseudo_200015779.csv \
input/denoised_valid_200015779.csv \
--resume model/01_efficientnet_b4_val15779_l2softmax_mixup_re_normalize_gray190_pseudo-efficientnet-b4-0054.model
--batch-size 96 \
--val-batch-size 96 \
--scale-size 210 \
--input-size 190 \
--epochs 80 \
--optimizer sgd \
--wd 1e-4 \
--base-lr 0.00001 \
--warmup-epochs 3 \
--lr-step-epochs 20,40,60,70 \
--rgb-mean 0.589 \
--rgb-std 0.364 \
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
--prefix 01_refine_efficientnet_b4_l2softmax_gray190
