from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from wrpsolver.bc.cunstomCnn import ResNet18,CustomCombinedExtractor
from stable_baselines3 import PPO
from typing import Callable

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value,1e-6)
    return func
class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int,  verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save('saved_model/decay')
            pass
        return True
    
def make_env(env_id, rank = 0, logFile = None,seed=0):
    def _init():
        env = GridWorldEnv(render=False)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v2'
    num_cpu = 16
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000)
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[256, 256],
        # share_features_extractor = False
    )
    model = PPO("MultiInputPolicy", use_expert = True, env = env, verbose=1,
                batch_size=2**10,n_steps=2**10,gamma=0.99,ent_coef=0.01,
                policy_kwargs = policy_kwargs,clip_range=0.1,learning_rate=linear_schedule(3e-5))
    # model.set_parameters("saved_model/decay")
    model.learn(total_timesteps=2e8,progress_bar=True,log_interval=1,callback=callback)


