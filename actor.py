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
        self.mean = tf.layers.dense(
            x, self.act_dim
        )

        logvar_speed = self.act_dim * 2.0
        log_vars = tf.get_variable(
            "log_vars", (logvar_speed, self.act_dim),
            tf.float32, tf.constant_initializer(0.0)
        )
        self.log_vars = tf.reduce_sum(log_vars, axis=0)

        self.sampled_act = self.mean + \
            tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=(self.act_dim,))

    def sampling(self, obses):
        feed_dict = {
            self.obs_ph: obses
        }
        return self.sess.run(self.sampled_act, feed_dict)

    def act(self, obses):
        feed_dict = {
            self.obs_ph: obses
        }
        return self.sess.run(self.mean, feed_dict)