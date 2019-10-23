#!/usr/bin/env python
# coding: utf-8

# usage
# $ python plot_recognition_results.py <target_pickle.dir> <target_image_dir>
# $ python plot_recognition_results.py test_nms030_tta7_5models_hard_prob input/test_images

import os
import sys
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import japanize_matplotlib

from PIL import Image
from util.functions import Rectangle


target_pickle_dir = sys.argv[1]
target_image_dir = sys.argv[2]


def unicodeToCharacter(codepoint):
    return chr(int(codepoint.replace('U+', '0x'), 16))

def drawBoxAndText(ax, label):
    codepoint, x, y, w, h = label
    x, y, w, h = int(x), int(y), int(w), int(h)
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=1)
    ax.add_patch(rect)
    ax.text(x+w+5, y+(h/2)+10, unicodeToCharacter(codepoint), color='r', size=24)
    return ax

for pickle_path in glob.glob(target_pickle_dir + '/*.pickle'):
    image_id = pickle_path.split('/')[-1].replace('.pickle', '')
    org_img_path = os.path.join(target_image_dir, image_id + '.jpg')

    with open(pickle_path, 'rb') as f:
        sub_info = pickle.load(f)

    width, height = sub_info['size']

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width/100, height/100))

    displayimg = Image.open(org_img_path).convert('RGB')
    ax.imshow(displayimg)

    for label, bbox in zip(sub_info['pred_labels'], sub_info['new_boxlist']):
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        drawBoxAndText(ax, (label, x, y, w, h))

    save_filename = os.path.join(target_pickle_dir, image_id + '_with_results.jpg')
    plt.savefig(save_filename, bbox_inches='tight')
    print("saved:", save_filename)
    plt.close()

