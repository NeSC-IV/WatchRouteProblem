from gym.envs.registration import register
from .gym_env import GridWorldEnv
register(
    id='IL/GridWorld-v0',
    entry_point='wrpsolver.bc:GridWorldEnv',
    max_episode_steps=300,
)