#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import numpy as np
import scipy.stats as stats

from util.functions import Rectangle, get_labels, chunks, score_page, get_center_point, print_result_score, broken_box_check, KModel, ensemble_boxes_idx, predict_image

mode = 'val'
nms_threshold = 0.30

save_dir_list = [
  'val_detector_060000_tta7_5models_hard_prob',
  'val_detector_100000_tta7_5models_hard_prob',
]
generate_dir = 'val_nms030_tta7_5models_hard_prob'

with open('util/class_names.pickle', 'rb') as f:
    class_names = pickle.load(f)


if mode == 'val':
    target_images = [line.rstrip() for line in open('input/val_images.list').readlines()]
else:
    target_images = [line.rstrip() for line in open('input/test_images.list').readlines()]
count = len(target_images)


os.makedirs(generate_dir, exist_ok=False)

for target_index in range(0, count):
    image_id = target_images[target_index]

    prob_list_list = []
    pred_labels_list = []
    bbox_score_list = []
    bbox_list = []
    for save_dir in save_dir_list:

        load_file = os.path.join(save_dir,image_id + '.pickle')
        with open(load_file, 'rb') as f:
            r = pickle.load(f)

        size = r['size']
        prob_list = r['prob_list']
        pred_labels = r['pred_labels']
        bbox_score = r['bbox_score']
        new_boxlist = r['new_boxlist']
        prob_list_list.extend(prob_list)
        pred_labels_list.extend(pred_labels)
        bbox_score_list.extend(bbox_score)
        bbox_list.extend(new_boxlist)

    # soft-nms
    if len(prob_list) > 0:
        _, _, target_index = ensemble_boxes_idx(np.array(bbox_list), np.array(prob_list_list), nms_threshold, removeAloneBox=False)
    else:
        target_index = np.array([])

    renew_prob_list = []
    renew_pred_labels = []
    renew_bbox_score = []
    renew_boxlist= []
    for i in target_index.tolist():
        renew_prob_list.append(prob_list_list[i])
        renew_pred_labels.append(pred_labels_list[i])
        renew_bbox_score.append(bbox_score_list[i])
        renew_boxlist.append(bbox_list[i])

    # update pickle
    renew = {}
    renew['size'] = size
    renew['prob_list'] = renew_prob_list
    renew['pred_labels'] = renew_pred_labels
    renew['bbox_score'] = renew_bbox_score
    renew['new_boxlist'] = renew_boxlist
    save_file = os.path.join(generate_dir, image_id + '.pickle')
    with open(save_file, 'wb') as nf:
        pickle.dump(renew, nf)

    print(".", end='')
    sys.stdout.flush()

print('done!', generate_dir)
