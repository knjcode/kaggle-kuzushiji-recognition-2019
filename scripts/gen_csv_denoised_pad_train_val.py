#!/usr/bin/env python
# coding: utf-8

# usage:
# python gen_csv_denoised_pad_train_val.py 200015779

import sys
import pandas as pd
import numpy as np

try:
    val_label = sys.argv[1]
except:
    print("specify book name for validation")
    sys.exit(1)

df_train = pd.read_csv('./input/train_characters.csv', header=None)
df_train.columns = ['Unicode', 'filepath']

uniq_char = df_train.Unicode.unique()


train_df_list = []
val_df_list = []

for i, cur_char in enumerate(uniq_char):
    cur_df = df_train[df_train.Unicode == cur_char]

    tmp_train = cur_df.drop(cur_df.index[cur_df.filepath.str.contains(val_label)])
    tmp_val = cur_df[cur_df.filepath.str.contains(val_label)]


    if len(tmp_val) == 0:
        # If there is no character of the specified book, random sample up to 20 copies from train
        val_count = int(len(tmp_train) * 0.10)
        if val_count > 20:
            cur_val = tmp_train.sample(20)
            tmp_train = tmp_train.drop(cur_val.index)
        else:
            # characters that occur 20 times or less are also copied to validation
            cur_val = cur_df
    else:
        cur_val = tmp_val

    if len(tmp_train) == 0:
        # Random samples up to 20 if there are no characters in the train
        # except for the specified book characters.
        train_count = int(len(tmp_val) * 0.10)
        if train_count > 20:
            cur_train = tmp_val.sample(20)
            cur_val = tmp_val.drop(cur_train.index)
        else:
            # characters that occur 20 times or less are also copied to train
            cur_train = cur_df
    else:
        cur_train = tmp_train

    train_df_list.append(cur_train)
    val_df_list.append(cur_val)

    if i % 100 == 0:
        print(".", end='')
        sys.stdout.flush()
print("preprocess done!")


train_df = pd.concat(train_df_list)
val_df = pd.concat(val_df_list)

print("postprocess for train data for class contains less than 100 images...")

# Oversample characters that appear less than 100 times more than 100 times
counter = train_df.Unicode.value_counts()
code_and_count = {}
for elem in train_df.Unicode.unique():
    if counter[elem] < 100:
        code_and_count[elem] = counter[elem]

add_train_df_list = []
for elem, count in code_and_count.items():
    multi_count = int(100 / count)
    for i in range(multi_count):
        add_train_df_list.append(train_df[train_df.Unicode == elem])

add_train_df = pd.concat(add_train_df_list)
train_df = pd.concat([train_df, add_train_df])

print("done!")

print("postprocess for validation data for class contains less than 20 images...")

# Oversample characters that appear less than 20 times more than 20 times
counter = val_df.Unicode.value_counts()
code_and_count = {}
for elem in val_df.Unicode.unique():
    if counter[elem] < 20:
        code_and_count[elem] = counter[elem]

add_val_df_list = []
for elem, count in code_and_count.items():
    multi_count = int(20 / count)
    for i in range(multi_count):
        add_val_df_list.append(val_df[val_df.Unicode == elem])

print("done!")

print("finalizing...")
add_val_df = pd.concat(add_val_df_list)
val_df = pd.concat([val_df, add_val_df])

train_df.to_csv(f'./input/denoised_train_{val_label}.csv', header=False, index=False)
val_df.to_csv(f'./input/denoised_valid_{val_label}.csv', header=False, index=False)

print("all done!")
