#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: configuration.py
@time: 2019-03-04 10:13
"""


class Config(object):
    def __init__(self):
        self.params = {}
        self.max_round = None
        self.cv_folds = None
        self.early_stop_round = None
        self.seed = None
        self.save_model_path = None

    def xgb_config(self):

        self.params = {'learning_rate': 0.01,
                       'max_depth': 27,
                       'eta': 1,
                       'silent': 1,
                       'objective': 'multi:softmax',
                       'num_class': 3}
        self.max_round = 2
        self.cv_folds = 10
        self.early_stop_round = 500
        self.seed = 3
        self.save_model_path = '../model/xgb/'

    def xgb_config_1(self):
        self.params = {
                       # "objective": "multi:softprob",
                       "objective": "multi:softmax",
                       "num_class": 3,
                       "eval_metric": "merror",
                       # "eval_metric": "auc",
                       "learning_rate": 0.001,
                       # 'n_estimators': 5000,
                       "max_depth": 6,
                       "min_child_weight": 10,
                       "subsample": 0.76,
                       "colsample_bytree": 0.95,
                       "alpha": 2e-05,
                       "scale_pos_weight": 1,
                       "silent": 1,
                       "eta": 1,
                       "gamma": 0.70,
                       "lambda": 10}
        self.max_round = 100
        self.cv_folds = None
        self.early_stop_round = 500
        self.seed = 3
        # self.save_model_path = '../model/xgb/xgb.bat'

    def lgb_config(self):
        self.params = {'task': 'train',
                       'boosting_type': 'gbdt',
                       'objective': 'multiclass',
                       'num_class': 3,
                       'metric': ['multi_error', 'multi_logloss'],
                       'metric_freq': 1,
                       # 'max_bin': 255,
                       'num_leaves': 31,
                       'max_depth': 20,
                       'learning_rate': 0.05,
                       'feature_fraction': 0.9,
                       'bagging_fraction': 0.95,
                       'bagging_freq': 5}

        self.max_round = 100
        self.cv_folds = 4
        self.early_stop_round = 30
        self.seed = 3
        self.save_model_path = 'model/lgb/lgb.txt'


conf = Config()

