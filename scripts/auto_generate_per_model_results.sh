#!/bin/bash

# generate 10 results from 2detection models and 5 classification models
# replace model epoch number to best validation accuracy epoch number

model1_epoch=60  # 0.9667 (val15779)
model2_epoch=69  # 0.9625 (val15779)
model3_epoch=80  # 0.9590 (val15779)
model4_epoch=82  # 0.9592 (val3076)
model5_epoch=90  # 0.9597 (val3076)

for iter in 060000 100000; do
  # for mode in val; do  # if you want to generate validation results uncomment this and comment out the line below
  for mode in test; do

    python scripts/gen_single_detector_and_single_model_results.py \
    models/${mode}_${iter}.pth \
    models/01_refine_efficientnet_b4_l2softmax_gray190-00${model1_epoch}.model \
    --mode ${mode} --scale-size 200 --input-size 190 --normalize org \
    --output-dir ${mode}_detector_${iter}_tta7_01_efficientnet_b4

    python scripts/gen_single_detector_and_single_model_results.py \
    models/${mode}_${iter}.pth \
    models/02_refine_resnet152_l2softmax_gray112-00${model2_epoch}.model \
    --mode ${mode} --scale-size 120 --input-size 112 --normalize auto \
    --output-dir ${mode}_detector_${iter}_tta7_02_resnet152

    python scripts/gen_single_detector_and_single_model_results.py \
    models/${mode}_${iter}.pth \
    models/03_refine_seresnext101_l2softmax_rgb112-00${model3_epoch}.model \
    --mode ${mode} --scale-size 120 --input-size 112 --normalize org \
    --output-dir ${mode}_detector_${iter}_tta7_03_seresnext101

    python scripts/gen_single_detector_and_single_model_results.py \
    models/${mode}_${iter}.pth \
    models/04_refine_seresnext101_l2softmax_rgb112-00${model4_epoch}.model \
    --mode ${mode} --scale-size 120 --input-size 112 --normalize org \
    --output-dir ${mode}_detector_${iter}_tta7_04_seresnext101

    python scripts/gen_single_detector_and_single_model_results.py \
    models/${mode}_${iter}.pth \
    models/05_refine_resnet152_l2softmax_rgb112-00${model5_epoch}.model \
    --mode ${mode} --scale-size 120 --input-size 112 --normalize org \
    --output-dir ${mode}_detector_${iter}_tta7_05_resnet152

  done
done
