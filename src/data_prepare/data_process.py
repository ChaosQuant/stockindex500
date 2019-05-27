#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: data_process.py
@time: 2019-05-24 08:59
"""
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing


class DataProcess():
    def __init__(self):

        self.factor_list = ['TOPEN', 'HIGH', 'LOW', 'TCLOSE', 'VOTURNOVER', 'VATURNOVER']
        self.training_data_set = []
        self.label_set = []

    def cal_return(self, data):
        for i in range(len(data)):
            i = 1 + i
            if data[i-1] == 0:
                data[i-1] = 0
                continue
            if i % 6 == 0:
                # VATURNOVER 处理
                data[i-1] = data[-1] / data[i-1] -1
            elif (i + 1) % 6 == 0:
                # VOTURNOVER 处理
                data[i-1] = data[-2] / data[i-1] -1
            else:
                # data[-3], -3的位置为TCLOSE
                data[i-1] = data[-3] / data[i-1] - 1
        return data

    def training_data_prepare(self, source_df, series_len=30, total_len=50):
        print('len_source_df: %s' % len(source_df))

        for index in source_df.index:
            print('index %s' % index)
            if index > len(source_df) - total_len:
                break
            # index从零开始
            tmp = source_df.loc[index: index + total_len - 1, :]
            print("len_tmp %s" % len(tmp))
            date = tmp['TRADEDATE'].iloc[-1]  # 将序列的最后一个日期作为当前样本的日期
            print('trade_date %s' % date)
            # 数据准备, 选取指定长度(series_len)的时间序列，
            training_data = tmp[self.factor_list].iloc[0: series_len, :].values.reshape(1, -1).tolist()
            print("training_data:\n %s" % training_data)
            self.cal_return(training_data[0]).append(date)
            print("training_data:\n %s" % training_data)
            # training_data[0].append(date)
            self.training_data_set += training_data
            # label prepare
            label, t_min, t_max, max_min, tmp = self.label_created(tmp['TCLOSE'])
            label_tmp = [label, t_min, t_max, max_min, tmp, date]
            print('a %s' % label_tmp)
            self.label_set.append(label_tmp)
            print('label: %s' % label_tmp[0])
            print('len_train_data %s' % len(training_data))
            print('------------------------>')

    def label_created(self, df):
        # aa = df[['TOPEN', 'HIGH', 'LOW', 'TCLOSE', 'VOTURNOVER', 'VATURNOVER']].values.reshape(-1)

        # df = df['TCLOSE']
        tclose_30 = df.values[29]  # 第30分钟的数据值
        tclose_50 = df.values[49]  # 第50分钟的数据值

        tclose_min = min(df.values[30:])  # 30分钟之后的数据，包括第三十分钟
        tclose_max = max(df.values[30:])
        # 相对于前一时间点的close， 跳到最低点的值
        t_min = tclose_min - tclose_30
        # 相对于前一时间点的close， 跳到最高点的值
        t_max = tclose_max - tclose_30
        # 最高点跳到最低点的值
        max_min = t_max - t_min
        tt = tclose_50 - tclose_30

        tmp = (tclose_50 / tclose_30 - 1.0) * 10000
        print('t_min: %s, t_max: %s, max_min: %s, tmp: %s' % (t_min, t_max, max_min, tt))

        if abs(tmp) - 0.23 <= 0:
            label_retrns = 0
        elif tmp > 0:
            label_retrns = 1
        else:
            label_retrns = 2
        return [label_retrns, t_min, t_max, max_min, tmp]


    def data_saver(self):
        training_data_df = pd.DataFrame(self.training_data_set)
        label_df = pd.DataFrame(self.label_set, columns=['label_retrns', 't_min', 't_max', 'max_min', 'tmp', 'date'])

        # 数据保存
        with open("train_df_50_pro_all.pkl", 'wb') as pk:
            # pickle.dump(training_data_set, pk)
            # pickle.dump(training_data_df, pk)
            pickle.dump(training_data_df, pk)

        with open("label_df_50_pro_all.pkl", 'wb') as pk:
            # pickle.dump(label_set, pk)
            # pickle.dump(label_df, pk)
            pickle.dump(label_df, pk)


if __name__ == '__main__':
    pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width', 1000)
    path = "/Users/li/PycharmProjects/stockindex500/src/data/LCY_INDEX_01MS_SH_000905.pkl"

    index_500_01ms = pd.read_pickle(path=path)
    # print(index_500_01ms.head())

    # 时间格式修改
    index_500_01ms = index_500_01ms[['TRADEDATE', 'TOPEN', 'HIGH', 'LOW', 'TCLOSE', 'VOTURNOVER', 'VATURNOVER']]
    print('shape_of_index_500_01ms: {}'.format(np.shape(index_500_01ms)))
    # print('index_500_01ms: {}'.format(index_500_01ms.head(50)))

    # tmp_data = index_500_01ms[:100]
    tmp_data = index_500_01ms

    datapress = DataProcess()
    datapress.training_data_prepare(tmp_data)

    # print(pd.DataFrame(datapress.training_data_set))

    datapress.data_saver()
