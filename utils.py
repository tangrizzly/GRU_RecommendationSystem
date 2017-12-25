#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 13/12/2017 11:55 PM

@author: Tangrizzly
"""

import numpy as np
import pandas as pd
from config import Config

config = Config(1)

def load_taobao_data(config):
    with open(config.dataset)as f:
        data = f.readlines()

    buytimes = []
    buy_different = []
    user_id = []
    buys = []
    time_stamps = []
    whole_times = []

    for line in data[1:]:
        # line = re.sub('[\[|\]|\s++]', '', line)
        odom = line.split(' ')
        buytimes.append(int(odom[0]))
        buy_different.append(float(odom[1]))
        user_id.append(int(odom[2]))

        buys.append(list(map(int, odom[3].split(','))))
        time_stamps.append(list(map(int, odom[4].split(','))))
        whole_times = np.append(whole_times, list(map(int, odom[4].split(','))))

    unique_time_stamps = np.unique(np.array(whole_times))
    time_split_point = whole_times[int(np.size(unique_time_stamps) * config.split)-1]

    x_train = []
    x_test = []
    # Max_item_no = 0

    for i in np.arange(len(buys)):
        xi_train = [i for (i, j) in zip(buys[i], time_stamps[i]) if j <= time_split_point]
        xi_test = [i for (i, j) in zip(buys[i], time_stamps[i]) if j > time_split_point]
        _, idx = np.unique(xi_test, return_index=True)
        if len(idx) != 0 and len(xi_train) != 0:
            # print(i)
            x_train.append(xi_train)
            # Max_item_no = np.max([Max_item_no, np.max(xi_train), np.max(xi_test)])
            xi_test_dd = [xi_test[index] for index in sorted(idx)]
            x_test.append(xi_test_dd)

    x_train_pd = pd.DataFrame(data=x_train)
    x_test_pd = pd.DataFrame(data=x_test)
    Max_item_no = np.nanmax([x_train_pd.iloc[:, 1].values, x_test_pd.iloc[:, 1].values])

    return x_train_pd, x_test_pd, Max_item_no



