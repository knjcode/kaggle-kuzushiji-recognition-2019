#!/usr/bin/env python
# coding: utf-8

from PIL import Image

full_image_size_dict = {}

train_list = [elem.rstrip() for elem in open('input/train_images.list')]
for image_id in train_list:
    img = Image.open('input/train_images/' + image_id + '.jpg')
    full_image_size_dict[image_id] = img.size

test_list = [elem.rstrip() for elem in open('input/test_images.list')]
for image_id in test_list:
    img = Image.open('input/test_images/' + image_id + '.jpg')
    full_image_size_dict[image_id] = img.size

import pickle
with open('input/full_image_size_dict.pickle', 'wb') as f:
    pickle.dump(full_image_size_dict, f)

