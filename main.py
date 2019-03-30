import gym

import tensorflow as tf

from actor import Actor
from critic import Critic
from model import DynamicModel
from controller import MPC

class runner(object):
    def __init__(self, env, model, actor, critic, controller):
        self.env = env
        self.model = model
        self.actor = actor
        self.critic = critic
        self.controller = controller

    def run(self, episodes=200):
        for e in range(episodes):
            obs = env.reset()
            done = False
            while done:
                act = self.controller.step(obs)
                next_obs, rew, done, _ = env.step(act)
                self.model.collect(obs, act, next_obs)
            
            self.model.train()

if __name__ == "__main__":
    env = gym.make("Ant-v1")
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


