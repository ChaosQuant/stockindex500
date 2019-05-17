#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: m1_xgb.py
@time: 2019-03-04 09:36
"""

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import time
import pickle
import argparse
import numpy as np
import xgboost as xgb
from src.stacking.configuration import conf
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
# import matplotlib as plt


class XGBooster(object):
    def __init__(self, args):
        self.xgb_params = args.params
        self.num_boost_round = args.max_round
        self.cv_folds = args.cv_folds
        self.early_stop_round = args.early_stop_round
        self.seed = args.seed
        self.save_model_path = args.save_model_path
        self.xgb_params.update({'silent': 1, 'objective': 'multi:softmax', 'num_class': 3})

    def fit(self, x_train, y_train):
        xgb_start = time.time()
        if self.cv_folds is not None:
            print('cross_validation。。。。')
            d_train = xgb.DMatrix(x_train, label=y_train)
            cv_rounds = self._kfold(d_train)
            print('cv_result %s' % cv_rounds)
            print('type_cv_result %s' % type(cv_rounds))
            min_error = cv_rounds['test-merror-mean'].min()
            self.best_round = cv_rounds[cv_rounds['test-merror-mean'].isin([min_error])].index[0]
            self.best_score = min_error
            self.best_model = xgb.train(self.xgb_params, d_train, self.best_round)

        else:
            print('non_cross_validation。。。。')
            x_train, x_vaild, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_valid = xgb.DMatrix(x_vaild, label=y_valid)
            watchlist = [(d_train, "train"), (d_valid, "valid")]
            self.best_model = xgb.train(params=self.xgb_params,
                                        dtrain=d_train,
                                        num_boost_round=self.num_boost_round,
                                        evals=watchlist)
            self.best_round = self.best_model.best_iteration
            self.best_score = self.best_model.best_score
            cv_rounds = None

        print('spend time :' + str((time.time() - xgb_start)) + '(s)')
        return self.best_score, self.best_round, cv_rounds, self.best_model

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.best_model.predict(dpred)

    def _kfold(self, dtrain):
        cv_rounds = xgb.cv(self.xgb_params, dtrain,
                           num_boost_round=self.num_boost_round,
                           nfold=self.cv_folds,
                           seed=self.seed,
                           verbose_eval=True,
                           metrics={'merror', 'mlogloss'},
                           early_stopping_rounds=self.early_stop_round,
                           show_stdv=False)
        return cv_rounds

    def plot_feature_importances(self):
        feat_imp = pd.Series(self.best_model.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    def get_params(self, deep=True):
        return self.xgb_params

    def set_params(self, **params):
        self.xgb_params.update(params)

    def save_model(self, model_path=None):
        # now = time.strftime('%Y-%m-%d %H:%M')
        model_name = 'xgboost_{}.bat'.format(now)
        if model_path:
            joblib.dump(self.best_model, model_path + model_name)
        else:
            joblib.dump(self.best_model, self.save_model_path + model_name)

    def load_model(self, model_path=None):
        if model_path is None and self.save_model_path is None:
            print('model load error')
            exit()
        if model_path:
            bst_model = joblib.load(model_path)
        else:
            bst_model = joblib.load(self.save_model_path)
        return bst_model


def cal_acc(test_data, pre_data):
    pass


def xgb_predict(model, x_test, y_test, save_result_path=None):
    d_test = xgb.DMatrix(x_test)
    if conf.params['objective'] == "multi:softmax":
        y_pred = model.predict(d_test)
        print('shape_of_pre: %s' % y_pred)
        # 如果输入数据y_test为array
        # result = y_test.reshape(1, -1) == y_pred
        # 如果输入数据y_test为dataframe
        result = y_test.values.reshape(1, -1) == y_pred
        print('the accuracy:\t', float(np.sum(result)) / len(y_pred))
    else:
        # 输出概率
        y_pred_prob = model.predict(d_test)
        y_pred = y_pred_prob
    if save_result_path:
        df_reult = pd.DataFrame(x_test)
        df_reult['y_test'] = y_test
        df_reult['result'] = y_pred
        df_reult.to_csv(save_result_path, index=False)


def run_cv(x_train, x_test, y_train, y_test):
    x_train = x_train
    conf.xgb_config()
    tic = time.time()
    data_message = 'X_train.shape={}, X_test.shape = {}'.format(np.shape(x_train), np.shape(x_test))
    print(data_message)
    xgb = XGBooster(conf)
    best_auc, best_round, cv_rounds, best_model = xgb.fit(x_train, y_train)
    print('Training time cost {}s'.format(time.time() - tic))
    xgb.save_model()
    result_message = 'best_auc = {}, best_round = {}'.format(best_auc, best_round)
    print(result_message)

    # now = time.strftime('%Y-%m-%d %H:%M')
    result_saved_path = '../result/result_{}-{:.4f}.csv'.format(now, best_auc)
    xgb_predict(best_model, x_test, y_test, save_result_path=result_saved_path)


now = time.strftime('%Y-%m-%d %H:%M')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--result_save_path', type=str, default='../model/xgboost/')
    # parser.add_argument('--model_save_path', type=str, default='../result/xgboost/')

    # 输入数据为array格式
    # with open('../data_prepare/train.pkl', 'rb') as pk:
    #     train_dataset = pickle.load(pk)
    #     train_dataset = np.array(train_dataset)
    #     print("len_train_dataset %s" % len(train_dataset))
    #
    # with open('../data_prepare/label.pkl', 'rb') as pk:
    #     label_dataset = pickle.load(pk)
    #     label_dataset = np.array(label_dataset)
    #     print("len_label_dataset %s" % len(label_dataset))
    #
    # label = []
    # for i in label_dataset:
    #     if i == -1:
    #         label.append(2)
    #     else:
    #         label.append(i)
    #
    # label = np.array(label)
    # x_train, x_test, y_train, y_test = train_test_split(train_dataset[:30000], label_dataset[:30000], test_size=0.2, random_state=100)
    # run_cv(x_train, x_test, y_train, y_test)

    # 输入数据为dataframe格式

    with open('../data_prepare/train_df.pkl', 'rb') as pk:
        train_dataset_df = pickle.load(pk)

    with open('../data_prepare/label_df.pkl', 'rb') as pk:
        label_dataset_df = pickle.load(pk)

    x_train, x_test, y_train, y_test = train_test_split(train_dataset_df[:30000], label_dataset_df[:30000], test_size=0.2, random_state=10000, shuffle=True)
    print('x_train_pre: %s' % x_train.head())
    print('y_train_pre: %s' % y_train.head())
    print('x_test_pre: %s' % x_test.head())
    print('y_test_pre: %s' % y_test.head())

    # 数据统计用
    x_test.to_csv('../result/x_test_{}.csv'.format(now))
    y_test.to_csv('../result/y_test_{}.csv'.format(now))

    # 样本预处理
    x_train = x_train.drop([174], axis=1)
    x_test = x_test.drop([174], axis=1)
    y_train = y_train.drop(['t_min', 't_max', 'max_min', 'tmp', 'date'], axis=1)
    y_test = y_test.drop(['t_min', 't_max', 'max_min', 'tmp', 'date'], axis=1)

    print('x_train: %s' % x_train.head())
    print('y_train: %s' % y_train.head())
    print('x_test: %s' % x_test.head())
    print('y_test: %s' % y_test.head())

    # 模型训练
    run_cv(x_train, x_test, y_train, y_test)

