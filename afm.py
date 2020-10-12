#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/12 1:46 下午
# @Author  : Roamoin
# @File    : afm.py

import tensorflow as tf

from utils.common import LinearModel, EmbedModel


class AttentionalFactorizationMachine(tf.keras.Model):
    def __init__(self, feature_cards, factor_dim, attention_size, name='attentional_factorization_machine'):
        super().__init__(name=name)
        self.num_features = len(feature_cards)
        self.factor_dim = factor_dim
        self.linear = LinearModel(feature_cards, name=name + '/linera_model')
        self.embedding = EmbedModel(feature_cards, factor_dim, name=name + '/embedding')
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Dense(attention_size, name=name + '/attention_hidden'),
            # (none, num_interactions, attention_size)
            tf.keras.layers.ReLU(name=name + '/attention_relu'),  # (none, num_interactions, attention_size)
            tf.keras.layers.Dense(1, name=name + '/attention_logits'),  # (none, num_interactions, 1)
            tf.keras.layers.Softmax(1, name=name + '/attention_score')  # (none, num_interactions, 1)
        ])

    def call(self, inputs):
        batch_size = int(tf.shape(inputs)[0])
        num_interactions = self.num_features * (self.num_features - 1) // 2
        line_out = self.linear(inputs)  # (none, 1)

        factor = self.embedding(inputs)  # (none, num_features, factor_dim)
        factor_i = tf.tile(tf.expand_dims(factor, 1),
                           [1, self.num_features, 1, 1])  # (none,num_features, num_features, factor_dim )
        factor_j = tf.tile(tf.expand_dims(factor, 2),
                           [1, 1, self.num_features, 1])  # (none,num_features, num_features, factor_dim )
        interactions = tf.multiply(factor_i, factor_j)  # (none,num_features, num_features, factor_dim )

        mask = tf.ones(interactions.shape[:-1])  # (none, num_features, num_features)
        mask = tf.cast(tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0),
                       dtype=tf.bool)  # #(none, num_features, num_features)
        interactions = tf.boolean_mask(interactions, mask)  # (none*num_interactions, factor_dim)

        interactions = tf.reshape(interactions, [batch_size, num_interactions,
                                                 self.factor_dim])  # (none, num_interactions, factor_dim)
        attention_scores = self.attention(interactions)  # (none, num_interactions, 1)

        interactions = tf.reduce_sum(interactions, axis=-1, keep_dims=True)  # (none, num_interactions, 1)
        attended = tf.multiply(interactions, attention_scores)  # (none, num_interactions, 1)

        return line_out + tf.reduce_sum(attended, axis=1)  # (none, 1)


if __name__ == '__main__':
    x = [[1, 0, 3],
         [1, 0, 3],
         [1, 0, 3]]
    feature_cards = [3, 4, 5]
    factor_dim = 2
    attention_size = 3
    model = AttentionalFactorizationMachine(feature_cards, factor_dim, attention_size)
    result = model(x)
    assert result.shape == (3, 1)
