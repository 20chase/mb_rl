import numpy as np
import tensorflow as tf


class Actor(object):
    def __init__(self, sess, obs_ph, act_dim):
        self.sess = sess
        self.obs_ph = obs_ph
        self.act_dim = act_dim

        self._build_net()

    def _build_net(self):
        x = tf.layers.dense(
            self.obs_ph, 64, activation=tf.nn.tanh
        )
        x = tf.layers.dense(
            x, 64, activation=tf.nn.tanh
        )
        x = tf.layers.dense(
            x, self.act_dim
        )