#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from glob import glob
from multiprocessing import Pool


df_train = pd.read_csv('./input/train.csv')
output_dir = './input/denoised_cropped_images_pad/'
padding_rate = 0.05

os.makedirs(output_dir, exist_ok=True)

target_files = glob("./input/denoised_train/*.png")


def save_cropped_character(filepath):
    filename = filepath.split("/")[-1]
    image_id = filename.split(".")[0]

    elem = df_train.values[df_train["image_id"] == image_id].flatten()

    try:
        labels = elem[1]
        labels = np.array(str(labels).split(' ')).reshape(-1, 5)
    except ValueError:
        return

    img = Image.open(filepath).convert('RGBA')
    width, height = img.size

    for i, (codepoint, x, y, w, h) in enumerate(labels):
        x, y, w, h = int(x), int(y), int(w), int(h)
        x_pad = np.rint(w * padding_rate)
        y_pad = np.rint(h * padding_rate)
        xmin = max(x - x_pad, 0)
        ymin = max(y - y_pad, 0)
        xmax = min(x + w + x_pad, width)
        ymax = min(y + h + y_pad, height)

        img_crop = img.crop((xmin, ymin, xmax, ymax))
        target_dir = output_dir + str(codepoint)
        os.makedirs(target_dir, exist_ok=True)
        img_crop.save(target_dir + '/' + image_id + '_' + str(i) + '.png')

    return


process_num = 32
p = Pool(process_num)
p.map(save_cropped_character, target_files)
p.close()
