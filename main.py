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
    config = Config(1)
    if not os.path.exists("models"):
        os.mkdir("models")
    train_set, test_set, train_length, test_length, max_item_no, whole_items = load_taobao_data(config)
    sess = tf.Session()
    gru = taobao(config, sess, max_item_no, whole_items, train_set, test_set, train_length, test_length)
    gru.train()
    sess.close()