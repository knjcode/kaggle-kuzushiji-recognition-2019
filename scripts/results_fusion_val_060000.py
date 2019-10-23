#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import scipy.stats as stats

from util.functions import Rectangle, get_labels, chunks, score_page, get_center_point, print_result_score, broken_box_check, KModel, ensemble_boxes, predict_image

mode = 'val'
ensemble_mode = 'hard_prob'

save_dir_list = [
  'val_detector_060000_tta7_01_efficientnet_b4',
  'val_detector_060000_tta7_02_resnet152',
  'val_detector_060000_tta7_03_seresnext101',
  'val_detector_060000_tta7_04_seresnext101',
  'val_detector_060000_tta7_05_resnet152',
]

generate_dir = 'val_detector_060000_tta7_5models_hard_prob'

with open('util/class_names.pickle', 'rb') as f:
    class_names = pickle.load(f)


if mode == 'val':
    target_images = [line.rstrip() for line in open('input/val_images.list').readlines()]
else:
    target_images = [line.rstrip() for line in open('input/test_images.list').readlines()]
count = len(target_images)

os.makedirs(generate_dir)

for target_index in range(0, count):
    image_id = target_images[target_index]

    prob_list_list = []
    pred_labels_list = []
    for save_dir in save_dir_list:
        load_file = os.path.join(save_dir,image_id + '.pickle')
        with open(load_file, 'rb') as f:
            r = pickle.load(f)
        size = r['size']
        prob_list = r['prob_list']
        pred_labels = r['pred_labels']
        bbox_score = r['bbox_score']
        new_boxlist = r['new_boxlist']
        prob_list_list.append(prob_list)
        pred_labels_list.append(pred_labels)

    if ensemble_mode == 'hard_prob':
        mode_labels, count_labels = stats.mode(np.array(pred_labels_list), axis=0)
        if len(mode_labels) > 0:
            pred_labels = mode_labels[0]
            count_labels = count_labels[0]
            prob_list = [0] * len(pred_labels)
            for cur_pred_labels, cur_prob_list in zip(pred_labels_list, prob_list_list):
                for i, (use_pred_label, cur_pred_label, cur_prob, count) in enumerate(zip(pred_labels, cur_pred_labels, cur_prob_list, count_labels)):
                    if count > 1:
                        if use_pred_label == cur_pred_label:
                            if cur_prob >= prob_list[i]:
                                prob_list[i] = cur_prob
                    else:
                        # 結果がバラけた場合は一番 probが高い文字を選ぶ
                        if cur_prob >= prob_list[i]:
                            prob_list[i] = cur_prob
                            pred_labels[i] = cur_pred_label
        else:
            pred_labels = []
            prob_list = []
    else:
        raise

    # update pickle
    r['prob_list'] = prob_list
    r['pred_labels'] = pred_labels
    save_file = os.path.join(generate_dir, image_id + '.pickle')
    with open(save_file, 'wb') as nf:
        pickle.dump(r, nf)

print('done!')

