#!/bin/bash

# generate 10 results from 2detection models and first 5 classification models
# replace model epoch number to best validation accuracy epoch number

model1_epoch=61  # 0.9588 (val15779)
model2_epoch=54  # 0.9618 (val15779)
model3_epoch=54  # 0.9610 (val15779)
model4_epoch=54  # 0.9476 (val3076)
model5_epoch=77  # 0.9466 (val3076)

for iter in 060000 100000; do
for mode in val test; do

python scripts/gen_single_detector_and_multi_model_results.py \
models/${mode}_${iter}.pth \
models/01_efficientnet_b4_val15779_l2softmax_mixup_re_normalize_gray190-efficientnet-b4-00${model1_epoch}.model,\
models/02_resnet152_val15779_l2softmax_mixup_re_gray112-resnet152-00${model2_epoch}.model,\
models/03_seresnext101_val15779_l2softmax_mixup_re_rgb112-se_resnext101_32x4d-00${model3_epoch}.model,\
models/04_seresnext101_val3076_l2softmax_icap_re_rgb112-se_resnext101_32x4d-00${model4_epoch}.model,\
models/05_resnet152_val3076_l2softmax_icap_re_rgb112-resnet152-00${model5_epoch}.model \
--mode ${mode} \
--scale-size 200,120,120,120,120 \
--input-size 190,112,112,112,112 \
--normalize org,auto,org,org,org \
--output-dir ${mode}_detector_${iter}_tta7_first_5models_soft_prob

done
done
