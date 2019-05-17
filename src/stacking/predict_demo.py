#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: predict_demo.py
@time: 2019-05-17 10:37
"""

from src.stacking.m1_xgb import *


if __name__ == '__main__':
    conf.xgb_config()
    xgb_model = XGBooster(conf)
    # data, shape = [N * M]
    x_test = pd.read_csv('../result/x_test_2019-05-17 11:22.csv')
    x_test = x_test.drop(['174'], axis=1)

    # load model
    model_path = "../model/xgb/xgboost_2019-05-17 11:14.bat"
    bst_model = xgb_model.load_model(model_path)

    # predict, shape = [N * 1]
    pre = xgb_model.predict(bst_model, x_test.values)
    print(pre)

