#!/bin/bash

find input/pseudo_images -type f > input/pseudo_images_path.csv
cut -d'/' -f3 input/pseudo_images_path.csv > input/pseudo_images_labels.csv
paste -d',' input/pseudo_images_labels.csv input/pseudo_images_path.csv > input/pseudo_images.csv
cat input/denoised_train_200015779.csv input/pseudo_images.csv > input/denoised_train_pseudo_200015779.csv
cat input/denoised_train_200003076.csv input/pseudo_images.csv > input/denoised_train_pseudo_200003076.csv
