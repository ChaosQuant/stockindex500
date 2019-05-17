#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: data_reader.py
@time: 2019-04-08 14:27
"""
import pysnooper
import pandas as pd
import numpy as np
import pickle
from src.utils import data_source

pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width', 1000)

path = "/Users/li/PycharmProjects/stockindex500/src/data/LCY_INDEX_01MS_SH_000905.pkl"
#
# engine_sqlserver = data_source.GetDataEngine("FACTOR")
# # engine_sqlserver = data_source.GetDataEngine("DNDS")
#
# @pysnooper.snoop()
# def read_stock_code(sheet_name):
#     """
#     读取股票名称和股票代码
#     :param sheet_name:
#     :return:
#     """
#     sql = "SELECT * FROM dbo.%s" % sheet_name
#     # sql = "SELECT * FROM dbo.%s" % sheet_name
#     result = pd.read_sql(sql, engine_sqlserver)
#     return result
#
#
# stock_df = read_stock_code('LCY_INDEX_01MS_SH_000905')
# # stock_df = read_stock_code('TQ_SK_BASICINFO')
# stock_df.to_pickle(path="/Users/li/PycharmProjects/stockindex500/src/train_dataset/LCY_INDEX_01MS_SH_000905.pkl")
# print(stock_df.head())


index_500_01ms = pd.read_pickle(path=path)
# print(index_500_01ms.head())

# 时间格式修改
index_500_01ms = index_500_01ms[['TRADEDATE', 'TOPEN', 'HIGH', 'LOW', 'TCLOSE', 'VOTURNOVER', 'VATURNOVER']]
print('shape_of_index_500_01ms: {}'.format(np.shape(index_500_01ms)))
print('index_500_01ms: {}'.format(index_500_01ms.head(50)))

# tmp_data = index_500_01ms[:100]
tmp_data = index_500_01ms


# @pysnooper.snoop(prefix='label_created:')
def label_created(df):
    # aa = df[['TOPEN', 'HIGH', 'LOW', 'TCLOSE', 'VOTURNOVER', 'VATURNOVER']].values.reshape(-1)

    # df = df['TCLOSE']
    tclose_30 = df.values[29]
    tclose_50 = df.values[49]

    tclose_min = min(df.values[30:])
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
    return label_retrns, t_min, t_max, max_min, tmp


training_data_set = []
label_set = []

for index in tmp_data.index:
    print("index %s" % index)
    if index > len(tmp_data) - 50:
        break
    tmp = tmp_data.loc[index: index + 49, :]
    print("len_tmp %s" % len(tmp))
    date = tmp['TRADEDATE'].iloc[-1]
    print('date %s' % date)
    training_data = tmp[['TOPEN', 'HIGH', 'LOW', 'TCLOSE', 'VOTURNOVER', 'VATURNOVER']].iloc[0:29, :].values.reshape(1, -1).tolist()
    print("training_data %s" % training_data)
    training_data[0].append(date)
    training_data_set += training_data

    print('training_data %s '% training_data)
    print(type(training_data))
    print(training_data)
    print(np.shape(training_data))

    label, t_min, t_max, max_min, tmp = label_created(tmp['TCLOSE'])
    a = [label, t_min, t_max, max_min, tmp, date]
    print('a %s' % a)
    label_set.append(a)
    print('label: %s' % label)
    print('len_train_data %s' % len(training_data))
    print('------------------------>')

# print(training_data_set)
training_data_df = pd.DataFrame(training_data_set)
print(type(training_data_set))
print(np.shape(training_data_set))
# training_data_df.to_csv('train.csv')

label_df = pd.DataFrame(label_set, columns=['label_retrns', 't_min', 't_max', 'max_min', 'tmp', 'date'])
# label_df.to_csv('label.csv')
# print(label_df)

# print(np.array(training_data_set))

# 数据保存
with open("train_df.pkl", 'wb') as pk:
    # pickle.dump(training_data_set, pk)
    pickle.dump(training_data_df, pk)

with open("label_df.pkl", 'wb') as pk:
    # pickle.dump(label_set, pk)
    pickle.dump(label_df, pk)


# aa['returns'] = aa['TCLOSE'].rolling(50).apply(label_created)




