#!/bin/bash

echo "Denoising and Ben's preprocessing for train and test images"
python scripts/denoising_and_bens_preprocessing.py train
python scripts/denoising_and_bens_preprocessing.py test

echo "Crop characters from preporcessed images"
python scripts/crop_characters_from_images.py

echo "Generate train_characters.csv"
find input/denoised_cropped_images_pad -type f > input/train_characters_path.csv
cut -d'/' -f3 input/train_characters_path.csv > input/train_characters_labels.csv
paste -d',' input/train_characters_labels.csv input/train_characters_path.csv > input/train_characters.csv

echo "Genrate train csv validation=200015779"
python scripts/gen_csv_denoised_pad_train_val.py 200015779

echo "Genrate train csv validation=200003076"
python scripts/gen_csv_denoised_pad_train_val.py 200003076

echo "Generate full_image_size_dict.pickle"
python scripts/gen_image_size_dict.py
