#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: __init__.py.py
@time: 2019-03-04 09:20
"""


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

# iris = datasets.load_iris()
# train_dataset = iris.train_dataset[:100]
# print(train_dataset.shape)
#
# label = iris.target[:100]
# print(label)
#
# train_x, test_x, train_y, test_y = train_test_split(train_dataset, label, random_state=0)
# print(train_x)

# a = pd.DataFrame({'a': [1,2,3,None],'b': [1,2,3,4], 'c': [1,2,3,4]})
#
# b = a[a.isnull().values == False]
#
# print(b)

import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

n = 2

time_array = datetime.strptime(str(20160229), "%Y%m%d")
time_array = time_array - timedelta(days=365) * n
date_time = int(datetime.strftime(time_array, "%Y%m%d"))
print(date_time)

# time_array = time.strptime(str(20160229), "%Y%m%d")
# year = time_array.tm_year - 1
# mon = time_array.tm_mon
# day = time_array.tm_mday
# date_time = str(year) + str(mon) + str(day)
# time_array = time.strptime(date_time, "%Y%m%d")
# date_time = int(time.strftime("%Y%m%d", time_array))
# print(date_time)