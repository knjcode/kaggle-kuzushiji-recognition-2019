import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import pandas as pd
import pickle

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class KuzushijiDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, train_ids, data_csv, unicode_translation, label_type='original', ext_type=None, mode='train', transforms=None):
        self.data_dir = data_dir
        with open(train_ids) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.image_size_dict = {}
        self.data_csv = data_csv
        self.unicode_translation = unicode_translation
        self.label_type = label_type
        self.ext_type = ext_type
        self.mode = mode
        self.transforms = transforms
        self.df_ut = pd.read_csv(unicode_translation)
        try:
            with open('input/full_image_size_dict.pickle', 'rb') as f:
                self.image_size_dict = pickle.load(f)
        except FileNotFoundError:
            self.image_size_dict = {}
        self.labels_dict, self.boxes_dict = self._preprocess_id_and_bbox(self.data_csv)

    def _preprocess_id_and_bbox(self, data_csv):
        df_train = pd.read_csv(data_csv)
        labels_dict = {}
        boxes_dict = {}

        id_set = set(self.ids)
        df_train = df_train.loc[df_train['image_id'].isin(id_set)]

        for image_id, labels in zip(df_train['image_id'], df_train['labels']):
            try:
                labels = np.array(labels.split(' ')).reshape(-1, 5)
                labels_list = []
                boxes_list = []
                for label in labels:
                    if self.label_type == 'original':
                        unicode_label = label[0]
                        label_index = self.df_ut[self.df_ut.Unicode == unicode_label].index[0]
                    else:
                        label_index = 0
                    bbox = [int(label[1]), int(label[2]), int(label[1])+int(label[3]), int(label[2])+int(label[4])]
                    labels_list.append(int(label_index) + 1)
                    boxes_list.append(bbox)
                labels_dict[image_id] = labels_list
                boxes_dict[image_id] = boxes_list
            except AttributeError:
                # Support training when no ground truth boxes are present
                width, height = self.image_size_dict[image_id]
                labels_dict[image_id] = [0]
                boxes_dict[image_id] = [[0, 0, 0, 0]]
        return labels_dict, boxes_dict

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        if self.ext_type:
            filename = image_id + '.' + self.ext_type
        else:
            filename = image_id + '.jpg'
        image = Image.open(self.data_dir + '/' + filename).convert("RGB")

        if self.mode == 'train':
            target = self.get_groundtruth(idx)
            target = target.clip_to_image(remove_empty=True)
        else:
            # generate dummy labels
            image_id = self.ids[idx]
            width, height = self.image_size_dict[image_id]
            target = BoxList([[0,0,0,0]], (width, height), mode="xyxy")
            target.add_field("labels", torch.tensor([0]))
            target.add_field("difficult", torch.tensor([0]))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # return the image, the boxlist and the idx in your dataset
        return image, target, idx

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, idx):
        image_id = self.ids[idx]

        width, height = self.image_size_dict[image_id]
        try:
            boxes = self.boxes_dict[image_id]
            labels = torch.tensor(self.labels_dict[image_id])
        except KeyError:
            boxes = [[0, 0, 0, 0]]
            labels = torch.tensor([0])
        target = BoxList(boxes, (width, height), mode="xyxy")
        difficults = torch.tensor([0]*len(labels))
        target.add_field("labels", labels)
        target.add_field("difficult", difficults)
        return target

    def get_img_info(self, idx):
        image_id = self.ids[idx]
        try:
            width, height = self.image_size_dict[image_id]
        except KeyError:
            filename = image_id + '.png'
            image = Image.open(self.data_dir + '/' + filename).convert("RGB")
            self.image_size_dict[image_id] = image.size
            width, height = image.size
        return {"height": height, "width": width}

    def map_class_id_to_class_name(self, class_id):
        return self.df_ut.Unicode[class_id]
