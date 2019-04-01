import numpy as np

from utils import timed


class MPC(object):
    def __init__(self, env, model, actor, critic):
        self.env = env
        self.model = model
        self.actor = actor
        self.critic = critic

        self.gamma = 0.95
        self.horizon = 20
        self.traj_num = 1000

    def _look_forward(self, batch_obs, batch_acts, env_num):
        xposbefores = [ob[0] for ob in batch_obs]
        rewards = np.zeros((len(batch_obs)))
        for _ in range(self.horizon):
            batch_obs = self.model.predict(batch_obs, batch_acts)
            xposafters = [ob[0] for ob in batch_obs]
            rewards += self.reward_function(
                xposbefores, xposafters, batch_acts)
            xposbefores = xposafters.copy()
            batch_acts = self.sample(env_num * self.traj_num)

        batch_rewards = np.expand_dims(rewards, axis=1)
        seq_rewards = self._batch_to_seq(batch_rewards)
        return seq_rewards

    def _batch_to_seq(self, batch):
        batch = np.asarray(batch)
        seq = []
        seq_num = batch.shape[0] / self.traj_num
        for i in range(int(seq_num)):
            seq.append(batch[i:(i+1)*self.traj_num, :])
        return seq

    def _seq_to_batch(self, seq):
        batch = []
        for s in seq:
            batch.extend(s)
        batch = np.asarray(batch)
        return batch

    def reward_function(self, xposbefores, xposafters, batch_acts):
        reward_ctrl = - 0.1 * np.sum(np.square(batch_acts), axis=1)
        reward_run = (np.asarray(xposafters) - np.asarray(xposbefores)) / 0.05
        reward = reward_ctrl + reward_run
        return reward
   
    def step(self, obs):
        env_num = len(obs)
        batch_acts = self.sample(env_num * self.traj_num)
        seq_acts = self._batch_to_seq(batch_acts)
        seq_obs = [np.tile(ob, (self.traj_num, 1)) for ob in obs]
        batch_obs = self._seq_to_batch(seq_obs)
        seq_rewards = self._look_forward(batch_obs, batch_acts, env_num)
        final_acts = []
        for rew, act in zip(seq_rewards, seq_acts):
            final_acts.append(act[np.argmax(rew)])
        return final_acts

    def random_step(self, obs):
        acts = self.sample(len(obs))
        return acts

    def sample(self, num):
        return np.random.uniform(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            size=(num, self.env.action_space.shape[0])).astype(
                self.env.action_space.dtype)

        