import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gym
import mb_env

import numpy as np
import tensorflow as tf

from actor import Actor
from critic import Critic
from model import DynamicModel
from controller import MPC
from environment import create_multi_env
from utils import timed

class Runner(object):
    def __init__(self, env, model, actor, critic, controller):
        self.env = env
        self.model = model
        self.actor = actor
        self.critic = critic
        self.controller = controller

        self.gamma = 0.99
        self.lam = 0.95

    def _init_mb(self):
        self.mb_obs = []
        self.mb_acts = []
        self.mb_next_obs = []

    def _collect_mb(self, obs, acts, next_obs):
        self.mb_obs.extend(obs)
        self.mb_acts.extend(acts)
        self.mb_next_obs.extend(next_obs)

    def random_run(self):
        self._init_mb()
        obs = env.reset()
        for _ in range(300):
            acts = self.controller.random_step(obs)
            next_obs, rews, dones, _ = self.env.step(acts)
            self._collect_mb(obs, acts, next_obs)

        self.model.collect(self.mb_obs,
                           self.mb_acts,
                           self.mb_next_obs)
        self.model.train(6000)
        
    def mpc_run(self):
        self._init_mb()
        obs = env.reset()
        mb_rews = []
        for _ in range(200):
            acts = self.controller.step(obs)
            next_obs, rews, dones, _ = self.env.step(acts)
            self._collect_mb(obs, acts, next_obs)
            mb_rews.append(rews)

        mb_rews = np.asarray(mb_rews)

        self.model.collect(self.mb_obs,
                           self.mb_acts,
                           self.mb_next_obs)
        self.model.train(6000)

        return np.mean(np.sum(mb_rews, axis=0))
            

if __name__ == "__main__":
    env = create_multi_env("MbHalfCheetah-v0", 16)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(graph=graph, config=config)

    model = DynamicModel(sess, obs_dim, act_dim)
    actor = Actor(sess, model.obs_ph, act_dim)
    critic = Critic(sess, model.obs_ph)
    controller = MPC(env, model, actor, critic)

    sess.run(tf.global_variables_initializer())
    runner = Runner(env, model, actor, critic, controller)
    runner.random_run()
    for e in range(100):
        print("episode {}: {}".format(
            e, runner.mpc_run()))
        if e % 10 == 0 and (e != 0):
            model.init_buffer()
    # runner.run()


