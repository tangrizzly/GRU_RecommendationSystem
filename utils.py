#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 13/12/2017 11:55 PM

@author: Tangrizzly
"""

import numpy as np


def load_taobao_data(config):
    with open(config.dataset)as f:
        data = f.readlines()

    buytimes = []
    buy_different = []
    user_id = []
    buys = []
    time_stamps = []
    whole_times = []
    whole_items = []

    for line in data[1:]:
        odom = line.split(' ')
        buytimes.append(int(odom[0]))
        buy_different.append(float(odom[1]))
        user_id.append(int(odom[2]))

        buys.append(list(map(int, odom[3].split(','))))
        whole_items.extend(list(map(int, odom[3].split(','))))
        time_stamps.append(list(map(int, odom[4].split(','))))
        whole_times.extend(list(map(int, odom[4].split(','))))

    # 按时间点切分数据。比如有10个月的数据，用前80%的做训练，就是前8个月训练，后2个月测试。
    # 并不是由size来决定时间点。

    stmp_min = min(whole_times)
    stmp_max = max(whole_times)
    time_split_point = int(stmp_min + (stmp_max - stmp_min) * config.split)

    # whole_times = list(sorted(set(whole_times)))
    # time_split_point = whole_times[int(np.size(whole_times) * config.split)-1]      # 这么写，不大准确。

    x_train = []
    x_test = []
    x_train_length = []
    x_test_length = []
    Max_item_no = 0

    for i in np.arange(len(buys)):
        xi_train = [i for (i, j) in zip(buys[i], time_stamps[i]) if j <= time_split_point]
        xi_test = [i for (i, j) in zip(buys[i], time_stamps[i]) if j > time_split_point]
        _, idx = np.unique(xi_test, return_index=True)
        if len(idx) > 0 and len(xi_train) > 1:  # train/test里只要有数据，就保留
            x_train.append(xi_train)
            x_train_length.append(len(xi_train))
            Max_item_no = np.max([Max_item_no, np.max(xi_train), np.max(xi_test)])
            xi_test_dd = [xi_test[index] for index in sorted(idx)]
            x_test.append(xi_test_dd)
            x_test_length.append(len(xi_test))

    all_items = set(whole_items)
    whole_num, item_num = len(whole_items), len(all_items)
    print('Use aliases to represent items ...')
    aliases_dict = dict(zip(all_items, range(1, item_num + 1)))
    train = [[aliases_dict[i] for i in buy] for buy in x_train]
    test = [[aliases_dict[i] for i in buy] for buy in x_test]

    return train, test, x_train_length, x_test_length, item_num
