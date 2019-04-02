import numpy as np
import tensorflow as tf


class DynamicModel(object):
    def __init__(self, sess, obs_dim, act_dim):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self._build_ph()
        self._build_net()
        self._build_opt()

        self.init_buffer()

    def init_buffer(self):
        self.ob_buffer = []
        self.act_buffer = []
        self.next_ob_buffer = []

    def _build_ph(self):
        self.obs_ph = tf.placeholder(
            tf.float32, [None, self.obs_dim],
            "obs_ph"
            )
        self.act_ph = tf.placeholder(
            tf.float32, [None, self.act_dim],
            "act_ph"
        )
        self.next_obs_ph = tf.placeholder(
            tf.float32, [None, self.obs_dim],
            "next_obs_ph"
        )

    def _build_net(self):
        net_in = tf.concat(
            [self.obs_ph, self.act_ph], axis=1)

        x = tf.layers.dense(
            net_in, 256, activation=tf.nn.relu)
        x = tf.layers.dense(
            x, 256, activation=tf.nn.relu)
        x = tf.layers.dense(
            x, self.obs_dim)
        
        self.pred_obs = x + self.obs_ph

    def _build_opt(self):
        self.loss = tf.losses.mean_squared_error(
            self.next_obs_ph, self.pred_obs
        )
        self.opt = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def _fit(self, obs, act, next_obs):
        feed_dict = {
            self.obs_ph: obs,
            self.act_ph: act,
            self.next_obs_ph: next_obs
        }
        return self.sess.run([self.loss, self.opt], feed_dict)[:-1]

    def collect(self, obses, acts, next_obses):
        self.ob_buffer.extend(obses)
        self.act_buffer.extend(acts)
        self.next_ob_buffer.extend(next_obses)
        
    def train(self, last_ind, batch_size=256, epochs=20):
        def to_np(arr):
            return np.asarray(arr)
        buffer_len = len(self.ob_buffer)
        if buffer_len - last_ind < 0:
            last_ind = buffer_len
        inds = np.arange(buffer_len-last_ind, buffer_len)
        total_loss = []
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, buffer_len, batch_size):
                end = start + batch_size
                if (end + batch_size) > buffer_len:
                    end = buffer_len
                mbinds = inds[start:end]
                slices = (to_np(arr)[mbinds] for arr in (self.ob_buffer,
                                                  self.act_buffer,
                                                  self.next_ob_buffer))
                total_loss.append(self._fit(*slices))
        
        print("model_loss", np.mean(total_loss))

    def predict(self, obses, acts):
        feed_dict = {
            self.obs_ph: obses,
            self.act_ph: acts
        }
        return self.sess.run(self.pred_obs, feed_dict)