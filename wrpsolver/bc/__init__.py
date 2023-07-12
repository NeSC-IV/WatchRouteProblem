from gym.envs.registration import register
# from .gym_env_hwc import GridWorldEnv
from .gym_env_hwc import GridWorldEnv
from .gym_env_hwc_100 import GridWorldEnv as GridWorldEnv_100
register(
    id='IL/GridWorld-v0',
    entry_point='wrpsolver.bc:GridWorldEnv',
    max_episode_steps=400,
)
register(
    id='IL/GridWorld-v1',
    entry_point='wrpsolver.bc:GridWorldEnv_100',
    max_episode_steps=400,
)