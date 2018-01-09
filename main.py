#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 25/12/2017 6:14 PM

@author: Tangrizzly
"""

from model2 import *
from config import Config
import os


if __name__ == "__main__":
    config = Config()
    if not os.path.exists("models"):
        os.mkdir("models")
    train_set, test_set, train_length, test_length, item_num = load_taobao_data(config)
    sess = tf.Session()
    gru = taobao(config, sess, item_num, train_set, test_set, train_length, test_length)
    gru.train()
    sess.close()