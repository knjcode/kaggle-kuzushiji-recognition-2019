#!/bin/bash

# If you have more GPUs, you can change the number of
# GPUs below to shorten the processing time.
NGPUS=2

# If you want to genrate prediction of validation dataset.
# Uncomment the following validation section

model060000="models/model_0060000.pth"
model100000="models/model_0100000.pth"

prediction_val_060000="models/val_060000.pth"
prediction_val_100000="models/val_100000.pth"
prediction_test_060000="models/test_060000.pth"
prediction_test_100000="models/test_100000.pth"


# # validation
# if [ ! -e $model060000 ]; then
#     echo "$model0600000 not found. Please download this file first."
# else
#     if [ ! -e $model100000 ]; then
#         echo "$model1000000 not found. Please download this file first."
#     else
#         # val 60000iter
#         if [ -e $prediction_val_060000 ]; then
#             echo "$prediction_val_060000 exists. Skip generate prediction."
#         else
#             python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
#             --ckpt "models/model_0060000.pth" \
#             --config-file "configs/kuzushiji/e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc.yaml" \
#             OUTPUT_DIR models/060000_val
#             mv models/060000_val/inference/kuzushiji_denoised_test_use_val_same_label/predictions.pth $prediction_val_060000
#         fi

#         # val 100000iter
#         if [ -e $prediction_val_100000 ]; then
#             echo "$prediction_val_100000 exists. Skip generate prediction."
#         else
#             python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
#             --ckpt "models/model_0100000.pth" \
#             --config-file "configs/kuzushiji/e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc.yaml" \
#             OUTPUT_DIR models/100000_val
#             mv models/100000_val/inference/kuzushiji_denoised_test_use_val_same_label/predictions.pth $prediction_val_100000
#         fi
#     fi
# fi


# test
if [ ! -e $model060000 ]; then
    echo "$model060000 not found. Please download this file first."
else
    if [ ! -e $model100000 ]; then
        echo "$model100000 not found. Please download this file first."
    else
        # test 60000iter
        if [ -e $prediction_test_060000 ]; then
            echo "$prediction_test_060000 exists. Skip generate prediction."
        else
            python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
            --ckpt ${model060000} \
            --config-file "configs/kuzushiji/e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc_test.yaml" \
            OUTPUT_DIR models/060000_test
            mv models/060000_test/inference/kuzushiji_denoised_test/predictions.pth $prediction_test_060000
        fi

        # test 100000iter
        if [ -e $prediction_test_100000 ]; then
            echo "$prediction_test_100000 exists. Skip generate prediction."
        else
            python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py \
            --ckpt ${model100000} \
            --config-file "configs/kuzushiji/e2e_faster_rcnn_R_101_C4_1x_2_gpu_voc_test.yaml" \
            OUTPUT_DIR models/100000_test
            mv models/100000_test/inference/kuzushiji_denoised_test/predictions.pth $prediction_test_100000
        fi

    fi
fi
