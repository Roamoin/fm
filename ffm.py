#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 11:58 上午
# @Author  : Roamoin
# @File    : ffm.py

import tensorflow as tf
from utils.common import LinearModel, FieldAwareEmbedModel


class FieldAwareFactorzationMachine(tf.keras.Model):
    def __init__(self, feature_cards, factor_dim, name='field_aware_factorzation_machine'):
        super().__init__(name=name)
        self.linear = LinearModel(feature_cards)
        self.embedding = FieldAwareFactorzationMachine(feature_cards, factor_dim)

    def call(self, inputs):
        linear_out = self.linear(inputs)  # (none, 1)

        factor_i = self.embedding(inputs)
        factor_j = tf.transpose(factor_i, [0, 2, 1, 3])  # (none, field, field, factor_dim)
        interactions = tf.reduce_sum(tf.multiply(factor_i, factor_j), -1)  # (none, field, field)
        interaction_out = tf.expand_dims(
            tf.reduce_sum(tf.linalg.band_part(interactions, 0, -1) - tf.linalg.band_part(interactions, 0, 0),
                          axis=(1, 2)), -1)  # (none,1)
        return linear_out + interaction_out


if __name__ == '__main__':
    x = [[1, 0, 3],
         [1, 0, 3],
         [1, 0, 3]]
    feature_cards = [3, 4, 5]
    factor_dim = 2
    model = FieldAwareFactorzationMachine(feature_cards, factor_dim)
    result = model(x)
    assert result.shape == (3, 1)
