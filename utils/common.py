#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 10:11 上午
# @Author  : Roamoin
# @File    : fm.py
import tensorflow as tf
import numpy as np


class LinearModel(tf.keras.Model):
    """
    y = wx+b
    feature_cards:
        [3, 4, 5] 包含3种特征
        性别: 男,女, 未知
        学历: 本科， 硕士， 博士， 其他
        年龄段: 0-20, 20-40, 40-60, 60-80, 80>
    inputs:
        [1, 0, 3]
        表示 女， 本科，60-80岁
    """

    def __init__(self, feature_cards, name='linear_model'):
        super().__init__(name=name)
        self.linear = tf.keras.layers.Embedding(sum(feature_cards), 1)
        self.bias = tf.random.uniform((1,))
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])))

    def call(self, inputs):
        x = inputs + tf.stop_gradient(self.offsets)
        return tf.reduce_sum(self.linear(x)) + self.bias


class EmbedModel(tf.keras.Model):
    def __init__(self, feature_cards, factor_dim, name='embedding_model'):
        super().__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(sum(feature_cards), factor_dim)
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])))

    def call(self, inputs):
        x = inputs + tf.stop_gradient(self.offsets)
        return self.embedding(x)


class FieldAwareEmbedModel(tf.keras.Model):
    def __init__(self, feature_cards, factor_dim, name='field_aware_embedding'):
        super().__init__(name=name)
        self.embeddings = [
            tf.keras.layers.Embedding(sum(feature_cards), factor_dim, name='{field}_embedding'.format(field=x)) for x, _ in enumerate(feature_cards)
        ]
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])))

    def call(self, inputs):
        x = tf.stop_gradient(input, self.offsets)
        outputs = [
            tf.expand_dims(embedding(x), 2) for embedding in self.embeddings
        ] # [(none, len(feature_cards), factor_dim), ....]长度为len(feature_cards)
        return tf.constant(outputs, 2)


class FullyConnectedNetwork(tf.keras.Model):
    def __init__(self, units, drop_out=.1, name='fcn'):
        super().__init__(name=name)
        self.layers = []
        for i, unit in enumerate(units):
            self.layers.append(tf.keras.layers.Dense(unit, name=name+'/fc{}'.format(i)))
            if drop_out>0:
                self.layers.append(tf.keras.layers.Dropout(drop_out, name=name+'dropout{}'.format(i)))
            if unit != 1:
                self.layers.append(tf.keras.layers.ReLU(name=name+'/active{}'.fromat(i)))
        self.model = tf.keras.Sequential(self.layers)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)