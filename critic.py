import numpy as np
import tensorflow as tf


class Critic(object):
    def __init__(self, sess, obs_ph):
        self.sess = sess
        self.obs_ph = obs_ph

        self._build_ph()
        self._build_net()
        self._build_opt()

    def _build_ph(self):
        self.ret_ph = tf.placeholder(
            tf.float32, [None], "ret_ph"
        )

    def _build_net(self):
        x = tf.layers.dense(
            self.obs_ph, 128, activation=tf.nn.tanh
            )
        x = tf.layers.dense(
            x, 64, activation=tf.nn.tanh
        )
        self.value = tf.layers.dense(
            x, 1
        )[:, 0]

    def _build_opt(self):
        self.loss = tf.reduce_mean(
            tf.square(self.value - self.ret_ph)
        )
        self.opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        
    def _fit(self, obs, ret):
        feed_dict = {
            self.obs_ph: obs,
            self.ret_ph: ret
        }
        return self.sess.run([self.loss, self.opt], feed_dict)[:-1]
    
    def train(self, obses, rets, batch_size=64, epochs=20):
        def to_np(arr):
            return np.asarray(arr)
        total_len = len(rets)
        inds = np.arange(total_len)
        total_loss = []
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, total_len, batch_size):
                end = start + batch_size
                if (end + batch_size) > total_len:
                    end = total_len
                mbinds = inds[start:end]
                slices = (to_np(arr)[mbinds] for arr in (obses, rets))
                total_loss.append(self._fit(*slices))
        
        return np.mean(total_loss)

    def predict(self, obses):
        feed_dict = {
            self.obs_ph: obses
        }
        return self.sess.run(self.value, feed_dict)


