#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 26/12/2017 10:16 AM

@author: Tangrizzly
"""
import tensorflow as tf

from utils import *


class taobao(object):
    def __init__(self, config, session, max_item_no):  # , x_train, x_test, x_train_length, x_test_length, Max_item_no):
        self.sess = session
        self.config = config
        self.mode = config.mode
        self.at_nums = config.at_nums
        self.latent_size = config.latent_size
        self.embedding_size = self.latent_size
        self.lr = config.alpha
        self.mini_batch = config.mini_batch
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.items_size = max_item_no+1
        self.keep_prob = config.keep_prob
        self.hidden_act = self.tanh
        self.final_activation = self.tanh
        self.loss_function = self.bpr
        self.layers = config.layers

        self.build()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def tanh(self, X):
        return tf.nn.tanh(X)

    def build(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size,], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.latent_size], name='rnn_state') for _ in
                      range(self.layers)]
        self.final_state = self.state
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = np.sqrt(6.0 / (self.items_size + self.latent_size))
            initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.items_size, self.latent_size], initializer=initializer)
            softmax_W = tf.get_variable('softmax_w', [self.items_size, self.latent_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.items_size], initializer=tf.constant_initializer(0.0))

            cell = tf.contrib.rnn.GRUCell(self.latent_size, activation=self.hidden_act)
            drop_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            stacked_cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * self.layers)

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        if self.mode == 'valid':
            # Use other examples of the minibatch as negative samples.
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
        else:
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = self.final_activation(logits)

        if not self.mode == 'valid':
            return

        optimizer = tf.train.AdamOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        self.train_op = optimizer.apply_gradients(gvs, global_step=self.global_step)

    def fit(self, data):
        self.error_during_train = False

        print('fitting model...')
        for epoch in range(self.epochs):
            epoch_cost = []
            state = [np.zeros([self.batch_size, self.latent_size], dtype=np.float32) for _ in range(self.layers)]
            idx_arr = np.arange(len(data))
            iters = np.arange(self.batch_size)
            finished = False
            while not finished:
                item_in = data.iloc[iters, :-1]
                item_out = data.iloc[iters, 1:]
                # prepare inputs, targeted outputs and hidden states
                fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                feed_dict = {self.X: item_in, self.Y: item_out}
                for j in range(self.layers):
                    feed_dict[self.state[j]] = state[j]

                cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                epoch_cost.append(cost)
                if np.isnan(cost):
                    print(str(epoch) + ':Nan error!')
                    self.error_during_train = True
                    return
                avgc = np.mean(epoch_cost)
                print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))

                iters = iters + self.batch_size
                if iters[-1] >= idx_arr[-1]:
                    finished = True

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)