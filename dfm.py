#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/12 2:21 下午
# @Author  : Roamoin
# @File    : dfm.py

import tensorflow as tf

from utils.common import LinearModel, EmbedModel, FullyConnectedNetwork


class DeepFM(tf.keras.Model):
    def __init__(self, feature_cards, factor_dim, hidden_size, dropout_rate=.1, name='deepfm'):
        super(DeepFM, self).__init__(name=name)
        self.linear = LinearModel(feature_cards)
        self.embedding = EmbedModel(feature_cards, factor_dim, name=name + '/embedding')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.fc = FullyConnectedNetwork(hidden_size, dropout_rate, name=name + '/fcn')

    def call(self, inputs, training=False):
        line_out = self.linear(inputs)  # (none, 1)

        factor = self.embedding(inputs)  # (none, len(feature_cards), factor_dim)

        sum_of_square = tf.pow(tf.reduce_sum(factor, axis=1), 2)  # (none, factor_dim)
        square_of_sum = tf.reduce_sum(tf.pow(factor, 2), axis=1)  # (none, factor_dim)
        interaction_out = 0.5 * tf.reduce_sum(sum_of_square - square_of_sum)

        fm_out = line_out + interaction_out
        fnn_out = self.fc(self.flatten(factor), training=training)
        return fm_out + fnn_out


if __name__ == '__main__':
    x = [[1, 0, 3],
         [1, 0, 3],
         [1, 0, 3]]
    feature_cards = [3, 4, 5]
    factor_dim = 2
    units = [5, 3, 1]
    drop_out = 0.1
    model = DeepFM(feature_cards, factor_dim, units, drop_out)
    result = model(x)
    assert result.shape == (3, 1)
