#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 26/12/2017 10:16 AM

@author: Tangrizzly
"""
import tensorflow as tf
from utils import *
import sys


class taobao(object):
    def __init__(self, config, session, max_item_no, whole_items, train_set, test_set, train_length, test_length):
        self.sess = session
        self.config = config
        self.at_nums = config.at_nums
        self.latent_size = config.latent_size
        self.embedding_size = self.latent_size
        self.alpha = config.alpha
        self.lmd = config.lmd
        self.batch_size_train = config.batch_size_train
        self.batch_size_test = config.batch_size_test
        self.epochs = config.epochs
        self.items_size = max_item_no + 1
        self.keep_prob = config.keep_prob
        self.whole_items = whole_items
        self.ckpt_path = config.ckpt_path
        self.train_set = train_set
        self.test_set = test_set
        self.train_length = train_length
        self.test_length = test_length
        self.save_per_epoch = config.save_per_epoch

    def build(self):
        self.item_input = tf.placeholder(tf.int32, [None, None], name='input')
        self.item_output = tf.placeholder(tf.int32, [None, None], name='output')
        self.item_neg_output = tf.placeholder(tf.int32, [None, None], name='negtive_output')
        sigma = np.sqrt(6.0 / (self.items_size + self.latent_size))
        initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
        self.embedding = tf.get_variable('embedding', [self.items_size, self.latent_size], initializer=initializer)
        self.item_input_embedded = tf.nn.embedding_lookup(self.embedding, self.item_input)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss = None
        self.rec10 = None
        self.max_10 = None

        with tf.variable_scope('gru_layer'):
            cell = tf.contrib.rnn.GRUCell(self.latent_size, activation=tf.tanh)
            self.cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.output, self.state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.item_input_embedded,
                                                        dtype=tf.float32)

        with tf.variable_scope('output'):
            cal_scores = tf.map_fn(lambda x: tf.matmul(x, self.embedding, transpose_b=True), self.output)
            # func = lambda x: tf.reduce_sum(
            #     tf.cast(tf.nn.in_top_k(x, tf.reshape(self.item_output, [-1]), 10), tf.float32))
            # self.rec10 = tf.map_fn(func, cal_scores)
            _, self.max_10 = tf.nn.top_k(cal_scores, k=10)

            self.item_output_embedded = tf.nn.embedding_lookup(self.embedding, self.item_output)
            self.y_p = tf.multiply(self.output, self.item_output_embedded)  # positive example
            self.item_neg_output_embedded = tf.nn.embedding_lookup(self.embedding, self.item_neg_output)
            self.y_n = tf.multiply(self.output, self.item_neg_output_embedded)  # positive example  # negative example
            tvars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tvars
                               if 'bias' not in v.name]) * self.lmd
            loss = tf.reduce_sum(-tf.log(tf.sigmoid(self.y_p - self.y_n)))
            self.loss = loss + lossL2

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss,
                                                                                  global_step=self.global_step)
        self.saver = tf.train.Saver()

    def train(self, mode=None, restore=False):
        if mode != "continue":
            print("Building the model...")
            self.build()
            self.sess.run(tf.global_variables_initializer())
        else:
            if restore:
                self.saver.restore(sess=self.sess, save_path=self.ckpt_path)

        print("Starting training...")
        print("%d steps per epoch." % (len(self.train_set) // self.batch_size_train))
        for epoch in range(self.epochs):
            loss_in_epoch = []
            rec10_in_epoch = []
            step = None
            for input_batch, input_length in self.minibatches(
                    inputs=self.train_set, input_length=self.train_length, batch_size=self.batch_size_train):
                # pad inputs
                x_batch, y_batch, y_n_batch = self.padding_sequence(input_batch)
                feed_dict = {
                    self.item_input: x_batch,
                    self.item_output: y_batch,
                    self.item_neg_output: y_n_batch
                }
                _, loss, step, max_10 = self.sess.run(
                    [self.train_op, self.loss, self.global_step, self.max_10], feed_dict=feed_dict)

                max10 = np.reshape(max_10, (-1, 10))
                target = np.reshape(y_batch, (-1))
                recall_rate = np.sum(np.isin(target, max10))/np.sum(input_length)

                loss_in_epoch.append(loss)
                rec10_in_epoch.append(recall_rate)

                print("Epoch %d, Step: %d, Loss: %.4f, Recall@10: %.4f" % (epoch, step, loss, recall_rate))

            print("Epoch %d, Step: %d, Loss: %.4f, Recall@10: %.4f" %
                  (epoch, step, np.mean(loss_in_epoch), np.mean(rec10_in_epoch)))

            if (epoch + 1) % self.save_per_epoch == 0:
                self.saver.save(self.sess, "models/bi-lstm-imdb.ckpt")

    def predict(self, restore=False):
        if restore:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(sess=self.sess, save_path=self.ckpt_path)

        print("Starting testing...")
        print("%d steps per epoch." % (len(self.test_set) // self.batch_size_test))
        loss_list = []
        rec10_list = []
        for input_batch, input_length in self.minibatches(
                inputs=self.test_set, input_length=self.test_length, batch_size=self.batch_size_test):
            # pad inputs
            x_batch, y_batch, y_n_batch = self.padding_sequence(input_batch)
            y_n_batch = np.random.choice(self.whole_items, batch_length).reshape(1, batch_length)
            feed_dict = {
                self.item_input: x_batch,
                self.item_output: y_batch,
                self.item_neg_output: y_n_batch
            }
            loss, max_10 = self.sess.run(
                [self.loss, self.max_10], feed_dict=feed_dict)

            max10 = np.reshape(max_10, (-1, 10))
            target = np.reshape(y_batch, (-1))
            recall_rate = np.sum(np.isin(target, max10)) / np.sum(input_length)

            loss_list.append(loss)
            rec10_list.append(recall_rate)
            print("Test finished on training set! Loss: %.4f, Acc: %.4f" % (np.mean(loss_list), np.mean(rec10_list)))

    def minibatches(self, inputs, input_length, batch_size, shuffle=False):
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], input_length[excerpt]

    def padding_sequence(self, inputs):
        batch_size = len(inputs)
        maxlen = np.max([len(i) for i in inputs])
        # length = np.sum([len(i) for i in inputs])
        x = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        y = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        y_n = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        for i, seq in enumerate(inputs):
            x[i][:len(seq[:-1])] = np.array(seq[:-1])
            y[i][:len(seq[1:])] = np.array(seq[1:])
            y_n[i] = np.random.choice(self.whole_items, maxlen - 1)
        return x, y, y_n  # , length
