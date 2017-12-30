#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 22/12/2017 9:00 AM

@author: Tangrizzly
"""

import os

class Config(object):
    def __init__(self, v):
        self.dataset = './user_buys.txt'
        self.split = 0.8
        self.at_nums = [10, 20, 30, 50]
        self.epochs = 30 if os.path.isfile('./user_buys.txt') else 50
        self.latent_size = 20
        self.alpha = 0.1
        self.lmd = 0.001
        self.mini_batch = 1  # 0:one_by_one,   1:mini_batch
        self.batch_size_train = 4 if self.mini_batch else 1  # size大了之后性能下降非常严重。one_by_one训练时该参数无效。
        self.batch_size_test = 768 if self.mini_batch else 1  # user*item矩阵太大，要多次计算。a5下亲测768最快。
        self.keep_prob = 1.0
        self.ckpt_path = "models"
        self.save_per_epoch = 10
