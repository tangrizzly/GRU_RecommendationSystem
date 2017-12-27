#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 22/12/2017 9:00 AM

@author: Tangrizzly

[1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
"""

import os


class Config(object):
    def __init__(self, v):
        # v = 0  # 写1就是valid, 写0就是test
        assert 0 == v or 1 == v  # no other case
        self.dataset = './user_buys.txt'
        self.mode = 'valid' if 1 == v else 'test'
        self.split = 0.8
        self.at_nums = [10, 20, 30, 50]
        self.epochs = 30 if os.path.isfile('./user_buys.txt') else 50
        self.latent_size = 20
        self.alpha = 0.1
        # self.lmd = 0.001
        self.mini_batch = 0  # 0:one_by_one,   1:mini_batch
        self.mvgru = 0  # 0:gru, 1:mv-gru, 2:mv-gru-2units, 3:mv-gru-con, 4:mv-gru-fusion
        self.batch_size_train = 4  # size大了之后性能下降非常严重。one_by_one训练时该参数无效。
        self.batch_size_test = 768  # user*item矩阵太大，要多次计算。a5下亲测768最快。
        self.batch_size = self.batch_size_train if 1 == v else self.batch_size_test
        self.keep_prob = 1.0
        self.layers = 1
