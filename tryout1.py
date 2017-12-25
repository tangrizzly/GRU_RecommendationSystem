#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 22/12/2017 8:32 AM

@author: Tangrizzly
"""

import numpy as np
import tensorflow as tf
from config import Config
from utils import *
import sys
import os


class taobao(object):
    def __init__(self, config, session, x_train, x_test, Max_item_no, train_length, test_lentgh):
        self.config = config
        self.mode = config.mode
        self.at_nums = config.at_nums
        self.latent_size = config.latent_size
        self.embedding_size = self.latent_size
        self.alpha = config.alpha
        self.lmd = config.lmd
        self.mini_batch = config.mini_batch
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.items_size = Max_item_no+1
        self.sess = session
        self.keep_prob = config.keep_prob

        self.x_train = x_train
        self.x_test = x_test

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def build(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.batch_maxlen = tf.placeholder(dtype=tf.int32, name="batch_maxlen")
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, name="output_keep_prob")
        self.item_input = tf.placeholder(dtype=tf.int32, name="item_input")
        self.item_output = tf.placeholder(dtype=tf.int32, name="item_output")
        self.embedding = tf.Variable(tf.random_uniform([self.items_size, self.embedding_size], -0.5, 0.5),
                                     dtype=tf.float32,
                                     name="item_embedding")
        self.encoder_input_embedded = tf.nn.embedding_lookup(self.embedding, self.item_input)
        softmax_W = tf.Variable(tf.random_uniform([self.items_size, self.embedding_size], -0.5, 0.5),
                                     dtype=tf.float32,
                                     name="softmax_W")
        softmax_b = tf.Variable(tf.zeros([self.items_size, self.embedding_size]),
                                     dtype=tf.float32,
                                     name="softmax_b")

        # define gru with dropout
        cell = tf.contrib.rnn.GRUCell(self.latent_size)
        self.cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)

        with tf.variable_scope("GRU") as scope:
            # [Batch size x time steps x features]
            output, state = tf.nn.dynamic_rnn(cell=self.cell,
                                              inputs=self.encoder_input_embedded,
                                              dtype=tf.float32)

        # sampled_W = tf.nn.embedding_lookup(softmax_W, self.item_output)
        # sampled_b = tf.nn.embedding_lookup(softmax_b, self.item_output)
        # logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
        # self.yhat = np.tanh(logits)
        # self.cost = self.bpr(self.yhat)

        if self.mode.equal('valid'):
            '''
            Use other examples of the minibatch as negative samples.
            '''
            sampled_W = tf.nn.embedding_lookup(softmax_W, self.item_output)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.item_output)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            self.yhat = tf.tanh(logits)
            self.cost = self.bpr(self.yhat)
            self.recall =
        else:
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = tf.tanh(logits)

        if not self.mode.equal('valid'):
            return

        optimizer = tf.train.AdamOptimizer(self.alpha)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        self.saver = tf.train.Saver()

    def fit(self, data):
        print('fitting model...')
        for epoch in range(self.epochs):
            epoch_cost = []
            for x in data.values:
                for i in range(len(x)-1):
                    in_idx = x[i]
                    out_idx = x[i+1]
                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: out_idx}
                    for j in xrange(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                start = start + minlen - 1
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions) - 1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]
                if len(mask) and self.reset_after_session:
                    for i in xrange(self.layers):
                        state[i][mask] = 0

            avgc = np.mean(epoch_cost)
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)

    def predict_next_batch(self, session_ids, input_item_ids, itemidmap, batch=50):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.

        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.

        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        batch : int
            Prediction batch size.

        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.

        '''
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1
            self.predict = True

        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:  # change internal states with session changes
            for i in xrange(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs}
        for i in xrange(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T
        return pd.DataFrame(data=preds, index=itemidmap.index)

