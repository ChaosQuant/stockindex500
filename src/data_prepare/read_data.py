#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: read_data.py.py
@time: 2019-04-28 11:22
数据分析模块
"""

import pickle
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width', 1000)


with open('train.pkl', 'rb') as pk:
    train_dataset_list = pickle.load(pk)

with open('label.pkl', 'rb') as pk:
    label_dataset_list = pickle.load(pk)

with open('train_df_50.pkl', 'rb') as pk:
    train_dataset = pickle.load(pk)

with open('label_df_50.pkl', 'rb') as pk:
    label_dataset = pickle.load(pk)

# print(train_dataset_list[:5])
# label = []
# for i in label_dataset_list:
#     if i == -1:
#         label.append(2)
#     else:
#         label.append(i)
# print(label[:300])

print(train_dataset.head(30))
print(label_dataset.head(30))
print(label_dataset.head(30)['label_retrns'].values.tolist())

print("训练集类型: {}".format(type(train_dataset)))
print("训练集结构: {}".format(np.shape(train_dataset)))
print("测试集类型: {}".format(type(label_dataset)))
print("测试集结构：{}".format(np.shape(label_dataset)))
print("测试集长度：{}".format(len(label_dataset)))

a1 = 0
a2 = 0
a3 = 0
a4 = 0

label_list = label_dataset['label_retrns'].values.tolist()
print(len(label_list))

for i in label_list:
    if i == 0:
        a1 += 1
    elif i == 1:
        a2 += 1
    elif i == 2:
        a3 += 1
    else:
        a4 += 1

print("'0'的个数{}， '1'的个数{}， '2'的个数{}, 其他：{}".format(a1, a2, a3, a4))
print("'0'的比例{}， '1'的比例{}， '2'的比例{}".format(a1/len(label_dataset), a2/len(label_dataset), a3/len(label_dataset), a4/len(label_dataset)))

# 训练集提取


