#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: m2_lgb.py
@time: 2019-03-04 11:03
"""

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import time
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.stacking.configuration import conf
from sklearn.model_selection import train_test_split


def lgb_fit(config, x_train, y_train):
    params = config.params
    print('params %s' % params)
    max_round = config.max_round
    cv_folds = config.cv_folds
    early_stop_round = config.early_stop_round
    seed = config.seed
    save_model_path = config.save_model_path

    if cv_folds is not None:
        print('cross_validation')
        d_train = lgb.Dataset(x_train, label=y_train)
        cv_result = lgb.cv(params,
                           d_train,
                           max_round,
                           nfold=cv_folds,
                           seed=seed,
                           verbose_eval=True,
                           metrics=['multi_error', 'multi_logloss'],
                           early_stopping_rounds=early_stop_round,
                           show_stdv=False)
        print('cv_result %s' % cv_result)
        print('type_cv_result %s' % type(cv_result))
        best_round = len(cv_result['multi_error-mean'])
        best_auc = cv_result['multi_error-mean'][-1]
        best_model = lgb.train(params, d_train, best_round)

    else:
        print('non_cross_validation')
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        watchlist = [d_train, d_valid]
        best_model = lgb.train(params, d_train, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None

    if save_model_path:
        pass
        # check_path(save_model_path)
        # best_model.save_model(save_model_path)
    return best_model, best_auc, best_round, cv_result


def lgb_predict(model, x_test, y_test, save_result_path=None):
    if conf.params['objective'] == "multiclass":
        y_pred = model.predict(x_test).argmax(axis=1)
        print(y_pred)
        print(y_test)
        result = y_test.reshape(1, -1) == y_pred
        print('the accuracy:\t', float(np.sum(result)) / len(y_pred))
    else:
        # 输出概率
        y_pred_prob = model.predict(x_test)
        y_pred = y_pred_prob
    if save_result_path:
        df_reult = pd.DataFrame()
        df_reult['result'] = y_pred
        df_reult.to_csv(save_result_path, index=False)


def run_feat_search(X_train, X_test, y_train, feature_names):
    pass


def run_cv(x_train, x_test, y_test, y_train):
    conf.lgb_config()
    tic = time.time()
    data_message = 'x_train.shape={}, x_test.shape={}'.format(x_train.shape, x_test.shape)
    print(data_message)
    # logger.info(data_message)
    lgb_model, best_score, best_round, cv_result = lgb_fit(conf, x_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_score={}'.format(best_round, best_score)
    # logger.info(result_message)
    print(result_message)

    # predict
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_score)
    # check_path(result_path)
    lgb_predict(lgb_model, x_test, y_test, save_result_path=None)


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
    data_message = 'X_train.shape={}, X_test.shape={}'.format(x_train.shape, x_test.shape)
    print(data_message)
    run_cv(x_train, x_test,y_test, y_train)


