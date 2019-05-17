#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: __init__.py.py
@time: 2019-03-04 14:03
"""
import time
import datetime
from src.stacking.configuration import conf
conf.xgb_config()
print(conf.params)
