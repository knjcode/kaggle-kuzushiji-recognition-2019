#!/bin/bash

./train.py \
input/denoised_train_200015779.csv \
input/denoised_valid_200015779.csv \
--model efficientnet-b4 \
--batch-size 128 \
--val-batch-size 256 \
--scale-size 210 \
--input-size 190 \
--epochs 80 \
--optimizer sgd \
--wd 1e-4 \
--base-lr 0.1 \
--warmup-epochs 5 \
--lr-step-epochs 20,40,60,70 \
--rgb-mean 0.589 \
--rgb-std 0.364 \
--random-resized-crop-scale 0.85,0.90 \
--random-resized-crop-ratio 0.875,1.14285714 \
--random-horizontal-flip 0.0 \
--random-vertical-flip 0.0 \
--random-rotate-degree 3 \
--jitter-brightness 0.1 \
--jitter-contrast 0.1 \
--jitter-saturation 0.1 \
--jitter-hue 0.05 \
--pca-noise 0 \
--workers 16 \
--from-scratch \
--grayscale \
--l2softmax \
--mixup \
--random-erasing \
--undersampling 2000 \
--drop-last \
--prefix 01_efficientnet_b4_val15779_l2softmax_mixup_re_normalize_gray190
