#!/bin/bash

./train.py \
input/denoised_train_pseudo_200003076.csv \
input/denoised_valid_200003076.csv \
--model se_resnext101_32x4d \
--batch-size 256 \
--val-batch-size 512 \
--scale-size 124 \
--input-size 112 \
--optimizer sgd \
--wd 1e-4 \
--base-lr 0.1 \
--warmup-epochs 5 \
--epochs 90 \
--cosine-annealing-t-max 86 \
--cosine-annealing-mult 1 \
--cosine-annealing-eta-min 1e-6 \
--random-resized-crop-scale 0.85,0.90 \
--random-resized-crop-ratio 0.875,1.14285714 \
--random-horizontal-flip 0.0 \
--random-vertical-flip 0.0 \
--random-rotate-degree 5 \
--jitter-brightness 0.25 \
--jitter-contrast 0.25 \
--jitter-saturation 0.25 \
--jitter-hue 0.125 \
--pca-noise 0.125 \
--random-grayscale-prob 0.2 \
--workers 16 \
--l2softmax \
--icap \
--icap-beta 1.0 \
--icap-prob 0.5 \
--random-erasing \
--undersampling 2000 \
--drop-last \
--prefix 04_seresnext101_val3076_l2softmax_icap_re_rgb112_pseudo
