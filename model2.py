#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on 26/12/2017 10:16 AM

@author: Tangrizzly
"""
import tensorflow as tf
from utils import *


class taobao(object):
    def __init__(self, config, session, item_num, train_set, test_set, train_length, test_length):
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
        self.item_num = item_num
        self.keep_prob = config.keep_prob
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
        self.item_output_mask = tf.placeholder(tf.float32, [None, None], name='output_mask')
        initializer = tf.random_uniform_initializer(minval=-0.5, maxval=0.5)
        self.embedding = tf.get_variable('embedding', [self.item_num+1, self.latent_size], initializer=initializer)
        self.item_input_embedded = tf.nn.embedding_lookup(self.embedding, self.item_input)
        self.batch_size = tf.placeholder(tf.float32, name='batch_size')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss = None
        self.max_10 = None
        self.output = None
        self.loss_all = None

        with tf.variable_scope('gru_layer') as gru:
            cell = tf.contrib.rnn.GRUCell(self.latent_size)
            self.cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.output, self.state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.item_input_embedded,
                                                        dtype=tf.float32)

        with tf.variable_scope('output'):
            cal_scores = tf.matmul(self.output[:, -1, :], self.embedding[1:], transpose_b=True)
            _, self.max_10 = tf.nn.top_k(cal_scores, k=50)

            self.item_output_embedded = tf.nn.embedding_lookup(self.embedding, self.item_output)
            self.item_neg_output_embedded = tf.nn.embedding_lookup(self.embedding, self.item_neg_output)
            self.prefr = tf.reduce_sum(self.output *
                                       (self.item_output_embedded - self.item_neg_output_embedded), 2)
            # loss = tf.reduce_sum(-tf.log(tf.sigmoid(self.prefr)) * self.item_output_mask)/self.batch_size

            loss = tf.reduce_mean(tf.reduce_sum((-tf.log(tf.sigmoid(self.prefr)))*self.item_output_mask, 1))

            wvars = [v for v in tf.trainable_variables() if 'weights' in v.name]
            bvars = [v / self.batch_size for v in tf.trainable_variables() if 'bias' in v.name]
            xvars = [self.item_output_embedded, self.item_neg_output_embedded]
            vars = wvars + bvars + xvars
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * self.lmd
            # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.lmd
            self.loss = lossL2 + loss

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)
        self.train_op = optimizer.minimize(loss, global_step=self.global_step)

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
        rec10_for_epoch = []
        for epoch in range(self.epochs):
            train_loss = []
            test_negs, train_negs = self.generate_neg_set()
            for input_batch, input_length, input_negs in self.minibatches(
                    inputs=self.train_set, input_length=self.train_length,
                    inputs_negs=train_negs, batch_size=self.batch_size_train):
                # pad inputs
                x_batch, y_batch, y_n_batch, y_batch_mask = self.padding_sequence(input_batch, input_negs)
                feed_dict = {
                    self.item_input: x_batch,
                    self.item_output: y_batch,
                    self.item_neg_output: y_n_batch,
                    self.item_output_mask: y_batch_mask,
                    self.batch_size: self.batch_size_train
                }

                _, step, loss = self.sess.run(
                    [self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
                # print("Epoch %d, Step: %d, Loss: %.4f" % (epoch, step, loss))
                train_loss.append(loss)

            hit = 0
            length = 0
            for input_batch, input_length, test_list, input_negs in self.minibatches_test(
                    inputs=self.train_set, input_length=self.train_length,
                    outputs=self.test_set, inputs_negs=train_negs,
                    batch_size=self.batch_size_test):
                x_batch, y_batch, y_n_batch, y_batch_mask = self.padding_sequence(input_batch, input_negs)
                feed_dict = {
                    self.item_input: x_batch,
                    self.item_output: y_batch,
                    self.item_neg_output: y_n_batch,
                    self.item_output_mask: y_batch_mask,
                    self.batch_size: self.batch_size_test
                }
                # test时为啥要算loss。。。。
                # user vector * all item vectors，得到每个user对所有items的评分，取前10个得分最大的items，并且按得分高低排序。
                max_10 = self.sess.run(
                    [self.max_10], feed_dict=feed_dict)
                for i in range(len(test_list)):
                    isin = np.apply_along_axis(lambda x: np.isin(x, max_10[i]), 0, test_list[i])
                    hit += np.sum(isin)
                    length += len(test_list)
                    # rec10.append(hit/length)    # 这种写法是算出每个用户的recall，再对所有的取平均。
                    # 更准确的写法是所有用户的hit加和、length加和，算总的recall。我程序里给的结果是这个的。
            rec10_for_epoch.append(hit/length)

            print(rec10_for_epoch)
            print("train loss %.4f"%(np.sum(train_loss)))

            if (epoch + 1) % self.save_per_epoch == 0:
                self.saver.save(self.sess, "models/bi-lstm-imdb.ckpt")

    def minibatches(self, inputs, input_length, inputs_negs, batch_size, shuffle=True):
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
                yield [inputs[i] for i in excerpt], [input_length[i] for i in excerpt], [inputs_negs[i] for i in excerpt]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
                yield inputs[excerpt], input_length[excerpt], inputs_negs[excerpt]

    def minibatches_test(self, inputs, outputs, inputs_negs, input_length, batch_size):
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], input_length[excerpt], outputs[excerpt], inputs_negs[excerpt]

    def padding_sequence(self, inputs, inputs_negs):
        batch_size = len(inputs)
        maxlen = np.max([len(i) for i in inputs])
        x = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        y = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        y_n = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        y_mask = np.zeros([batch_size, maxlen - 1], dtype=np.int32)
        for i, seq in enumerate(inputs):
            # assert len(seq) == len(inputs_negs[i])
            x[i][:len(seq[:-1])] = np.array(seq[:-1])   # 这里是对的，input是t时刻的item，算loss用t+1时刻的item
            y[i][:len(seq[1:])] = np.array(seq[1:])
            y_n[i][:len(seq[1:])] = np.array(inputs_negs[i][1:])
            y_mask[i][:len(seq[1:])] = 1
        return x, y, y_n, y_mask

    def generate_neg_set(self):
        test_negs = []
        train_negs = []
        for buy, pre in zip(self.train_set, self.test_set):
            neg_set = [i for i in range(1, self.item_num+1) if i not in buy or pre]
            # test_neg = np.random.choice(neg_set, len(pre))
            # test_negs.append(test_neg)
            train_neg = np.random.choice(neg_set, len(buy))
            train_negs.append(train_neg)
        return test_negs, train_negs
