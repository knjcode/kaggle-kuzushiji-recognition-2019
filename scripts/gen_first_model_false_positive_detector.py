#!/usr/bin/env python
# coding: utf-8

import lightgbm as lgb
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import numpy.random as rd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import optuna, os, uuid, pickle

df = pd.read_feather('val_nms030_tta7_first_5models_soft_prob.feather')
target = df.target

def target_reverse(value):
    return 1 - value

target = target.map(target_reverse)
tmp_df = df[df.columns.drop('target')]

# taken from https://www.kaggle.com/kenmatsu4/using-trained-booster-from-lightgbm-cv-w-callback
class ModelExtractionCallback(object):
    """Callback class for retrieving trained model from lightgbm.cv()
    NOTE: This class depends on '_CVBooster' which is hidden class, so it might doesn't work if the specification is changed.
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # Saving _CVBooster object.
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # Throw exception if the callback class is not called.
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # return Booster object
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # return list of Booster
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # return boosting round when early stopping.
        return self._model.best_iteration

rd_seed = 42
rd.seed(rd_seed)

X_train, y_train = tmp_df, target
lgb_train = lgb.Dataset(X_train, y_train)
extraction_cb = ModelExtractionCallback()

callbacks = [
    extraction_cb,
]

booster_name = 'models/booster_for_val_nms030_tta7_first_5models_soft_prob.pkl'

lgbm_params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'max_depth': 7,
    'max_bin': 255,
    'lambda_l1': 1.5464112458912599e-06,
    'lambda_l2': 5.346737781503549e-06,
    'num_leaves': 140,
    'feature_fraction': 0.8534057661739842,
    'bagging_fraction': 0.9376615592819334,
    'bagging_freq': 1,
    'min_child_samples': 72
 }


# Training settings
FOLD_NUM = 5
fold_seed = 7
folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)
# Fitting
ret = lgb.cv(params=lgbm_params,
               train_set=lgb_train,
               folds=folds,
               num_boost_round=200,
               verbose_eval = 10,
               early_stopping_rounds=50,
               callbacks=callbacks,
)

# Retrieving booster and training information.
proxy = extraction_cb.boosters_proxy
boosters = extraction_cb.raw_boosters
best_iteration = extraction_cb.best_iteration

with open(booster_name, "wb") as f:
    pickle.dump(boosters, f)

print("saved:", booster_name)
