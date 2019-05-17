#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: xgb_grid_search.py
@time: 2019-05-08 15:02
"""

import time
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV


if __name__ == '__main__':
    with open('../data_prepare/train.pkl', 'rb') as pk:
        train_dataset = pickle.load(pk)
        train_dataset = np.array(train_dataset)
        print("len_train_dataset %s" % len(train_dataset))

    with open('../data_prepare/label.pkl', 'rb') as pk:
        label_dataset = pickle.load(pk)
        label_dataset = np.array(label_dataset)
        print("len_label_dataset %s" % len(label_dataset))

    label = []
    for i in label_dataset:
        if i == -1:
            label.append(2)
        else:
            label.append(i)

    label = np.array(label)
    x_train, x_test, y_train, y_test = train_test_split(train_dataset[:30000], label[:30000], test_size=0.2, random_state=100)

    cv_params = {'n_estimators': [400, 500, 600, 700, 800]}

    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.train(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)


    optimized_GBM.fit(x_train, y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
