#!/bin/bash

./train.py \
input/denoised_train_pseudo_200003076.csv \
input/denoised_valid_200003076.csv \
--resume model/05_resnet152_val3076_l2softmax_icap_re_rgb112_pseudo-resnet152-0087.model \
--batch-size 96 \
--val-batch-size 96 \
--scale-size 124 \
--input-size 112 \
--epochs 90 \
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
--l2softmax \
--drop-last \
--onehot \
--refine \
--prefix 05_refine_resnet152_l2softmax_rgb112
