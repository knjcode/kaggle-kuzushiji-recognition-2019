#!/bin/bash

./train.py \
input/denoised_train_pseudo_200015779.csv \
input/denoised_valid_200015779.csv \
--model resnet152 \
--batch-size 512 \
--val-batch-size 1024 \
--scale-size 124 \
--input-size 112 \
--epochs 80 \
--optimizer sgd \
--wd 1e-4 \
--base-lr 0.1 \
--warmup-epochs 5 \
--lr-step-epochs 20,40,60,70 \
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
--prefix 02_resnet152_val15779_l2softmax_mixup_re_gray112_pseudo
