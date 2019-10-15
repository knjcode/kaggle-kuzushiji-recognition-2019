#!/bin/bash

# If you have more GPUs, you can change the number of
# GPUs below to shorten the training time.
NGPUS=2

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py \
--config-file "configs/kuzushiji/e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc.yaml" \
OUTPUT_DIR kuzushiji_recognition_R101_C4
