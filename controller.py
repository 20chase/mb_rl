import numpy as np


class MPC(object):
    def __init__(self, model, actor, critic):
        self.model = model
        self.actor = actor
        self.critic = critic

        self.horizon = 10
        self.traj_num = 10

    def _look_forward(self, obses, acts):
        for _ in range(self.horizon):
            obses = self.model.predict(obses, acts)
            acts = self.actor.act(obses)
        return self.critic.predict(obses)
        
    def step(self, obs):
        obses = [obs for _ in range(self.traj_num)]
        acts = self.actor.sampling(obses)
        values = self._look_forward(obses, acts)
        return acts[np.argmax(values)]
        

        


        
        