#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 25/12/2017 6:14 PM

@author: Tangrizzly
"""

# res = evaluation.evaluate_sessions_batch(gru, data, valid)
# print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))

from model import *
from config import Config


if __name__ == "__main__":
    config = Config(1)
    x_train_pd, x_test_pd, x_train_length, x_test_length, Max_item_no = load_taobao_data(config)
    sess = tf.Session()
    gru = taobao(config, sess, Max_item_no)
    if gru.mode == 'valid':
        # train a new model
        # tf.reset_default_graph()
        gru.fit(x_train_pd)
    else:
        gru.fit(x_test_pd)
    sess.close()