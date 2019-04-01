from gym.envs.registration import register

register(
    id='MbHalfCheetah-v0',
    entry_point='mb_env.envs:MbHalfCheetahEnv',
    max_episode_steps=1000
)
