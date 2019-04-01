import gym

import numpy as np
import tensorflow as tf

from actor import Actor
from critic import Critic
from model import DynamicModel
from controller import MPC

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
        self.mb_obses = []
        self.mb_acts = []
        self.mb_next_obses = []
        self.mb_rewards = []
        self.mb_values = []

    def _collect_mb(self, obs, act, next_obs, rew, value):
        self.mb_obses.append(obs)
        self.mb_acts.append(act)
        self.mb_next_obses.append(next_obs)
        self.mb_rewards.append(rew)
        self.mb_values.append(value)

    def _get_ret_adv(self, rews, values, last_value):
        step_len = len(rews)
        rets = np.zeros_like(rews)
        advs = np.zeros_like(rews)
        lastgaelam = 0        
        for t in reversed(range(step_len)):
            if t == step_len - 1:
                nextnonterminal = 0.0
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 
                nextvalues = values[t+1]
            delta = rews[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        rets = advs + values
        return rets, advs

    def run(self, episodes=200):
        for e in range(episodes):
            self._init_mb()
            obs = env.reset()
            done = False
            while not done:
                act = self.controller.step(obs)
                value = self.critic.predict([obs])[0]
                next_obs, rew, done, _ = env.step(act)
                self._collect_mb(obs, act, next_obs, rew, value)
            
            last_value = self.critic.predict([next_obs])[0]
            rets, advs = self._get_ret_adv(self.mb_rewards, 
                                           self.mb_values, last_value)
            self.critic.train(self.mb_obses, rets)
            self.model.collect(self.mb_obses, self.mb_acts, self.mb_next_obses)
            model_loss = self.model.train()
            print("episode: {} | rewards: {} | model_loss: {}".format(
                e, np.sum(self.mb_rewards), model_loss))

if __name__ == "__main__":
    env = gym.make("Pendulum-v0")
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
    controller = MPC(model, actor, critic)

    sess.run(tf.global_variables_initializer())
    runner = Runner(env, model, actor, critic, controller)
    runner.run()


