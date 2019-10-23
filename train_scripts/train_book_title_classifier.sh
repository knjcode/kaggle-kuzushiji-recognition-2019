#!/bin/bash

# generate csv as below
# find input/denoised_train -type f > input/book_train_path.csv
# cat input/book_train_path.csv | cut -d'/' -f3 | cut -d'_' -f1 | cut -d'-' -f1 | sed -e 's/brsk.*/brsk/g' | sed -e 's/hnsd.*/hnsd/g' | sed -e 's/umgy.*/umgy/g' > input/book_train_label.csv
# paste -d',' input/book_train_label.csv input/book_train_path.csv > input/book_train_tmp.csv

# cat book_train.csv
# 200015779,input/denoised_train/200015779_00053_2.png
# 200003076,input/denoised_train/200003076_00067_2.png
# 200004148,input/denoised_train/200004148_00074_1.png
# 100249476,input/denoised_train/100249476_00004_2.png
# 100249537,input/denoised_train/100249537_00033_2.png
# hnsd,input/denoised_train/hnsd009-003.png

# random split train:test 80%:20%
# shuf input/book_train_tmp.csv > input/book_train_shuf.csv
# head -n3105 input/book_train_shuf.csv > input/book_train.csv
# tail -n776 input/book_train_shuf.csv > input/book_valid.csv

./train.py \
input/book_train.csv \
input/book_valid.csv \
--model se_resnext50_32x4d \
--batch-size 64 \
--val-batch-size 128 \
--scale-size 256 \
--input-size 224 \
--optimizer sgd \
--wd 5e-4 \
--base-lr 0.1 \
--epochs 60 \
--warmup-epochs 5 \
--lr-step-epochs 15,30,45 \
--random-resized-crop-scale 0.85,1.0 \
--random-resized-crop-ratio 0.875,1.14285714 \
--random-horizontal-flip 0.0 \
--random-vertical-flip 0.0 \
--jitter-brightness 0.2 \
--jitter-contrast 0.2 \
--jitter-saturation 0.2 \
--jitter-hue 0.1 \
--pca-noise 0.1 \
--workers 16 \
--simple-label \
--drop-last \
--icap \
--icap-beta 2.0 \
--random-erasing \
--prefix book_title_classifier
