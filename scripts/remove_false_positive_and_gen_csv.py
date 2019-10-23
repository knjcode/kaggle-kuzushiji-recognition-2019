#!/usr/bin/env python
# coding: utf-8

import math
import os
import pickle
import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics

from util.functions import Rectangle, has_intersect, intersect, score_page, get_center_point, \
                            l2_distance, get_radian, get_nearest_box

save_dir = 'test_nms030_tta7_5models_hard_prob'

gather_info = True
cropping = False
expand_crop = True
padding_rate = 0.05
# crop_target_dir = 'pseudo_images'
# crop_prob = 1.0

target_images = [line.rstrip() for line in open('input/test_images.list').readlines()]
count = len(target_images)


def check_hiragana(label):
    codepoint = int(label.replace('U+', '0x'), 16)
    if 12352 <= codepoint <= 12447:
        return 1
    return 0

def is_first_in_second(a, b):
    return a[0] >= b[0] and b[2] >= a[2]  \
        and a[1] >= b[1] and b[3] >= a[3] 

def check_box(boxlist, size, prob_list):
    width, height = size

    broken_box_list = [0] * len(boxlist)
    inside_box_list = [0] * len(boxlist)
    has_box_list = [0] * len(boxlist)
    overlap_rate_list = [0.] * len(boxlist)

    for i, current_box in enumerate(boxlist):
        if broken_box_list[i] == 1:
            continue

        try:
            current_rect = Rectangle(*current_box)
        except ValueError:
            broken_box_list[i] = 1
            continue

        current_rect_overlap = 0.

        for j, target_box in enumerate(boxlist):
            try:
                target_rect = Rectangle(*target_box)
            except ValueError:
                borken_box_list[j] = 1
                continue

            if i == j:
                continue

            if is_first_in_second(current_box, target_box):
                inside_box_list[i] = 1
                has_box_list[j] = 1

            if has_intersect(current_rect, target_rect):
                overlap_rate = intersect(current_rect, target_rect).area() / current_rect.area()
                current_rect_overlap += overlap_rate

        overlap_rate_list[i] = current_rect_overlap

    return broken_box_list, inside_box_list, has_box_list, overlap_rate_list


def gen_info(prob, label, bbox, box_score, broken, overlap_rate, nearest_dict, new_boxlist, size, image_id):

    # 統計量調査
    w_list = []
    h_list = []
    area_list = []
    x_point_list = []
    y_point_list = []
    for xmin, ymin, xmax, ymax in new_boxlist:
        w = round(float(xmax - xmin))
        w_list.append(w)
        h = round(float(ymax - ymin))
        h_list.append(h)
        area_list.append(w*h)
        center_point = get_center_point((xmin, ymin, xmax, ymax))
        x_point_list.append(center_point[0])
        y_point_list.append(center_point[1])
    wl = pd.Series(w_list)
    hl = pd.Series(h_list)
    al = pd.Series(area_list)
    xl = pd.Series(x_point_list)
    yl = pd.Series(y_point_list)
    mean_area = al.mean()
    mean_width = wl.mean()
    mean_height = hl.mean()
    mean_x = xl.mean()
    mean_y = yl.mean()
    std_area = al.std()
    std_width = wl.std()
    std_height = hl.std()
    std_x = xl.std()
    std_y = yl.std()
    median_area = al.median()
    median_width = wl.median()
    median_height = hl.median()
    median_x = xl.median()
    median_y = yl.median()
    box_num = len(new_boxlist)

    try:
        nearest_box = new_boxlist[nearest_dict[0]['index']]
        nearest_width = round(float(nearest_box[2] - nearest_box[0]))
        nearest_height = round(float(nearest_box[3] - nearest_box[1]))
    except IndexError:
        nearest_width = np.nan
        nearest_height = np.nan
    try:
        nearest2_box = new_boxlist[nearest_dict[1]['index']]
        nearest2_width = round(float(nearest2_box[2] - nearest2_box[0]))
        nearest2_height = round(float(nearest2_box[3] - nearest2_box[1]))
    except IndexError:
        nearest2_width = np.nan
        nearest2_height = np.nan
    try:
        nearest3_box = new_boxlist[nearest_dict[2]['index']]
        nearest3_width = round(float(nearest3_box[2] - nearest3_box[0]))
        nearest3_height = round(float(nearest3_box[3] - nearest3_box[1]))
    except IndexError:
        nearest3_width = np.nan
        nearest3_height = np.nan
    try:
        nearest4_box = new_boxlist[nearest_dict[3]['index']]
        nearest4_width = round(float(nearest4_box[2] - nearest4_box[0]))
        nearest4_height = round(float(nearest4_box[3] - nearest4_box[1]))
    except IndexError:
        nearest4_width = np.nan
        nearest4_height = np.nan
    try:
        nearest5_box = new_boxlist[nearest_dict[4]['index']]
        nearest5_width = round(float(nearest5_box[2] - nearest5_box[0]))
        nearest5_height = round(float(nearest5_box[3] - nearest5_box[1]))
    except IndexError:
        nearest5_width = np.nan
        nearest5_height = np.nan

    try:
        nearest_radian = nearest_dict[0]['radian']
        nearest_distance = nearest_dict[0]['distance']
    except IndexError:
        nearest_radian = np.nan
        nearest_distance = np.nan
    try:
        nearest_radian2 = nearest_dict[1]['radian']
        nearest_distance2 = nearest_dict[1]['distance']
    except IndexError:
        nearest_radian2 = np.nan
        nearest_distance2 = np.nan
    try:
        nearest_radian3 = nearest_dict[2]['radian']
        nearest_distance3 = nearest_dict[2]['distance']
    except IndexError:
        nearest_radian3 = np.nan
        nearest_distance3 = np.nan
    try:
        nearest_radian4 = nearest_dict[3]['radian']
        nearest_distance4 = nearest_dict[3]['distance']
    except IndexError:
        nearest_radian4 = np.nan
        nearest_distance4 = np.nan
    try:
        nearest_radian5 = nearest_dict[4]['radian']
        nearest_distance5 = nearest_dict[4]['distance']
    except IndexError:
        nearest_radian5 = np.nan
        nearest_distance5 = np.nan

    center_point = get_center_point(bbox)
    sub_str = f"{label} {center_point[0]} {center_point[1]}"
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x_center, y_center = center_point
    current_info = {
        'image_id': image_id,
        'char': label,
        'char_score': prob,
        'is_hiragana': check_hiragana(label),
        'bbox': bbox,
        'bbox_score': box_score,
        # 'broken': broken,
        # 'inside': inside,
        # 'has_box': has_box,
        'overlap_rate': overlap_rate,
        'page_width': size[0],
        'page_height': size[1],
        # 'width': width,
        'width_page_rate': width / size[0],
        'width_mean_rate': width / mean_width,
        'width_std_rate': width / std_width if std_width else 0.,
        'width_median_rate': width / median_width,
        # 'height': height,
        'height_page_rate': height / size[1],
        'height_mean_rate': height / mean_height,
        'height_std_rate': height / std_height if std_height else 0.,
        'height_median_rate': height / median_height,
        # 'area': width * height,
        'area_page_rate': width * height / size[0] * size[1],
        'area_mean_rate': width * height / mean_area,
        'area_std_rate': width * height / std_area if std_area else 0,
        'area_median_rate': width * height / median_area,
        # 'nearest_width': nearest_width,
        'nearest_width_page_rate': nearest_width / size[0],
        'nearest_width_mean_rate': nearest_width / mean_width,
        'nearest_width_std_rate': nearest_width / std_width if std_width else 0,
        'nearest_width_median_rate': nearest_width / median_width,
        # 'nearest2_width': nearest2_width,
        'nearest2_width_page_rate': nearest2_width / size[0],
        'nearest2_width_mean_rate': nearest2_width / mean_width,
        'nearest2_width_std_rate': nearest2_width / std_width if std_width else 0,
        'nearest2_width_median_rate': nearest2_width / median_width,
        # 'nearest3_width': nearest3_width,
        'nearest3_width_page_rate': nearest3_width / size[0],
        'nearest3_width_mean_rate': nearest3_width / mean_width,
        'nearest3_width_std_rate': nearest3_width / std_width if std_width else 0,
        'nearest3_width_median_rate': nearest3_width / median_width,
        # 'nearest4_width': nearest4_width,
        'nearest4_width_page_rate': nearest4_width / size[0],
        'nearest4_width_mean_rate': nearest4_width / mean_width,
        'nearest4_width_std_rate': nearest4_width / std_width if std_width else 0,
        'nearest4_width_median_rate': nearest4_width / median_width,
        # 'nearest5_width': nearest5_width,
        'nearest5_width_page_rate': nearest5_width / size[0],
        'nearest5_width_mean_rate': nearest5_width / mean_width,
        'nearest5_width_std_rate': nearest5_width / std_width if std_width else 0,
        'nearest5_width_median_rate': nearest5_width / median_width,
        # 'nearest_height': nearest_height,
        'nearest_height_page_rate': nearest_height / size[0],
        'nearest_height_mean_rate': nearest_height / mean_height,
        'nearest_height_std_rate': nearest_height / std_height if std_height else 0,
        'nearest_height_median_rate': nearest_height / median_height,
        # 'nearest2_height': nearest2_height,
        'nearest2_height_page_rate': nearest2_height / size[0],
        'nearest2_height_mean_rate': nearest2_height / mean_height,
        'nearest2_height_std_rate': nearest2_height / std_height if std_height else 0,
        'nearest2_height_median_rate': nearest2_height / median_height,
        # 'nearest3_height': nearest3_height,
        'nearest3_height_page_rate': nearest3_height / size[0],
        'nearest3_height_mean_rate': nearest3_height / mean_height,
        'nearest3_height_std_rate': nearest3_height / std_height if std_height else 0,
        'nearest3_height_median_rate': nearest3_height / median_height,
        # 'nearest4_height': nearest4_height,
        'nearest4_height_page_rate': nearest4_height / size[0],
        'nearest4_height_mean_rate': nearest4_height / mean_height,
        'nearest4_height_std_rate': nearest4_height / std_height if std_height else 0,
        'nearest4_height_median_rate': nearest4_height / median_height,
        # 'nearest5_height': nearest5_height,
        'nearest5_height_page_rate': nearest5_height / size[0],
        'nearest5_height_mean_rate': nearest5_height / mean_height,
        'nearest5_height_std_rate': nearest5_height / std_height if std_height else 0,
        'nearest5_height_median_rate': nearest5_height / median_height,
        # 'nearest_area': nearest_width * nearest_height,
        'nearest_area_page_rate': nearest_width * nearest_height / size[0] * size[1],
        'nearest_area_mean_rate': nearest_width * nearest_height / mean_area,
        'nearest_area_std_rate': nearest_width * nearest_height / std_area if std_area else 0,
        'nearest_area_median_rate': nearest_width * nearest_height / median_area,
        # 'nearest2_area': nearest2_width * nearest2_height,
        'nearest2_area_page_rate': nearest2_width * nearest2_height / size[0] * size[1],
        'nearest2_area_mean_rate': nearest2_width * nearest2_height / mean_area,
        'nearest2_area_std_rate': nearest2_width * nearest2_height / std_area if std_area else 0,
        'nearest2_area_median_rate': nearest2_width * nearest2_height / median_area,
        # 'nearest3_area': nearest3_width * nearest3_height,
        'nearest3_area_page_rate': nearest3_width * nearest3_height / size[0] * size[1],
        'nearest3_area_mean_rate': nearest3_width * nearest3_height / mean_area,
        'nearest3_area_std_rate': nearest3_width * nearest3_height / std_area if std_area else 0,
        'nearest3_area_median_rate': nearest3_width * nearest3_height / median_area,
        # 'nearest4_area': nearest4_width * nearest4_height,
        'nearest4_area_page_rate': nearest4_width * nearest4_height / size[0] * size[1],
        'nearest4_area_mean_rate': nearest4_width * nearest4_height / mean_area,
        'nearest4_area_std_rate': nearest4_width * nearest4_height / std_area if std_area else 0,
        'nearest4_area_median_rate': nearest4_width * nearest4_height / median_area,
        # 'nearest5_area': nearest5_width * nearest5_height,
        'nearest5_area_page_rate': nearest5_width * nearest5_height / size[0] * size[1],
        'nearest5_area_mean_rate': nearest5_width * nearest5_height / mean_area,
        'nearest5_area_std_rate': nearest5_width * nearest5_height / std_area if std_area else 0,
        'nearest5_area_median_rate': nearest5_width * nearest5_height / median_area,
        # 'nearest_distance': nearest_dict[0]['distance'],
        'nearest_distance_page_width_rate': nearest_distance / size[0],
        'nearest_distance_page_height_rate': nearest_distance / size[1],
        # 'nearest2_distance': nearest_dict[1]['distance'],
        'nearest2_distance_page_width_rate': nearest_distance2 / size[0],
        'nearest2_distance_page_height_rate': nearest_distance2 / size[1],
        # 'nearest3_distance': nearest_dict[2]['distance'],
        'nearest3_distance_page_width_rate': nearest_distance3 / size[0],
        'nearest3_distance_page_height_rate': nearest_distance3 / size[1],
        # 'nearest4_distance': nearest_dict[3]['distance'],
        'nearest4_distance_page_width_rate': nearest_distance4 / size[0],
        'nearest4_distance_page_height_rate': nearest_distance4 / size[1],
        # 'nearest5_distance': nearest_dict[4]['distance'],
        'nearest5_distance_page_width_rate': nearest_distance5 / size[0],
        'nearest5_distance_page_height_rate': nearest_distance5 / size[1],
        'nearest_radian': nearest_radian,
        'nearest2_radian': nearest_radian2,
        'nearest3_radian': nearest_radian3,
        'nearest4_radian': nearest_radian4,
        'nearest5_radian': nearest_radian5,
        # 'x': x_center,
        # 'y': y_center,
        'x_mean_rate': x_center / mean_x,
        'y_mean_yrate': y_center /mean_y,
        'x_std_rate': x_center / std_x,
        'y_std_rate': y_center / std_y,
        'x_median_rate': x_center / median_x,
        'y_median_rate': y_center / median_y,
        'x_page_rate': x_center / size[0],
        'y_page_rate': y_center / size[1],
        'mean_area': mean_area,
        'mean_width': mean_width,
        'mean_height': mean_height,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'std_area': std_area,
        'std_width': std_width,
        'std_height': std_height,
        'std_x': std_x,
        'std_y': std_y,
        'median_area': median_area,
        'median_width': median_width,
        'median_height': median_height,
        'median_x': median_x,
        'median_y': median_y,
        'box_num': box_num,
    }
    return sub_str, current_info


def gen_csv_lgbm(prob_threshold, model_path, booster=False):

    if booster:
        with open(model_path, "rb") as fp:
            boosters = pickle.load(fp)
    else:
        with open(model_path, "rb") as fp:
            model = pickle.load(fp)

    after_score = []

    res = open('final_submission.csv', 'w')
    res.write('image_id,labels\n')

    write_count = 0

    for target_index in range(0, count):
        image_id = target_images[target_index]

        target_file = f'test_images/{image_id}.jpg'
        denoised_target_file = f'denoised_test/{image_id}.png'

        load_file = os.path.join(save_dir,image_id + '.pickle')
        with open(load_file, 'rb') as f:
            r = pickle.load(f)

        size = r['size']
        prob_list = r['prob_list']
        pred_labels = r['pred_labels']
        bbox_score = r['bbox_score']
        new_boxlist = r['new_boxlist']

        sub_info = []
        sub_list = []
        char_score_list = []
        box_score_list = []

        ## check box
        broken_box_list, inside_box_list, has_box_list, overlap_rate_list = check_box(new_boxlist, size, prob_list)

        ## check nearest box
        nearest_dict_list = get_nearest_box(new_boxlist)

        if cropping:
            orgimg = Image.open(denoised_target_file).convert('RGB')

        for i, (prob, label, bbox, box_score, broken, overlap_rate, nearest_dict) in \
            enumerate(zip(prob_list, pred_labels, new_boxlist, bbox_score, broken_box_list, overlap_rate_list, nearest_dict_list)):

            sub_str, current_info = gen_info(prob, label, bbox, box_score, broken, overlap_rate, nearest_dict, new_boxlist, size, image_id)

            sub_info.append(current_info)
            sub_list.append(sub_str)
            char_score_list.append(prob)
            box_score_list.append(box_score)

        sub_df = pd.DataFrame(sub_info)
        try:
            sub_df = sub_df.drop(['char', 'bbox', 'image_id'], axis=1)
        except KeyError:
            # sub_info is empty
            pass

        if len(sub_df) > 0:
            if booster:
                y_pred_list = []
                for booster in boosters:
                    y_pred_list.append(booster.predict(sub_df, num_iteration=booster.best_iteration))
                y_pred = np.average(y_pred_list, axis=0)
            else:
                y_pred = model.predict(sub_df, num_iteration=model.best_iteration)

            tmp_sub_list = []
            for current_info, sub, prob, char_score, box_score in zip(sub_info, sub_list, y_pred, char_score_list, box_score_list):
                (xmin, ymin, xmax, ymax) = current_info['bbox']

                if prob >= prob_threshold:
                    tmp_sub_list.append(sub)

            sub_list = tmp_sub_list

        else:
            sub_list = []

        sub_labels = ' '.join(sub_list)
        res.write(image_id.rstrip() + ',' + sub_labels + '\n')
        res.flush()
        write_count += 1
        print(".", end='')

    res.close()
    print('')
    print('write_count:', write_count)

gen_csv_lgbm(0.50, "models/booster_for_val_nms030_tta7_5models_hard_prob.pkl", booster=True)
