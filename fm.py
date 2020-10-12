#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 10:11 上午
# @Author  : Roamoin
# @File    : fm.py
import tensorflow as tf

from utils.common import LinearModel, EmbedModel


class FactorizationMachine(tf.keras.Model):
    def __init__(self, feature_cards, factor_dim, name='factorzation_machine'):
        super().__init__(name=name)
        self.embedding = EmbedModel(feature_cards, factor_dim)
        self.linear = LinearModel(feature_cards)

    def call(self, inputs):
        linear_out = self.linear(inputs)  # (none, 1)
        factor = self.embedding(inputs)  # (none, len(feature_cards), factor_dim)

        sum_of_squares = tf.pow(tf.reduce_sum(factor, 1), 2)  # (none, feator_dim)
        squares_of_sum = tf.reduce_sum(tf.pow(factor, 2), 1)  # (none, feator_dim)
        return linear_out + 0.5 * tf.reduce_sum(sum_of_squares - squares_of_sum, 1, keepdims=True)  # (none, 1)


if __name__ == '__main__':
    x = [[1, 0, 3],
         [1, 0, 3],
         [1, 0, 3]]
    feature_cards = [3, 4, 5]
    factor_dim = 2
    model = FactorizationMachine(feature_cards, factor_dim)
    result = model(x)
    assert result.shape == (3, 1)
