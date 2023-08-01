import shapely
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrpsolver.bc.gym_env_hwc_100_pp_dict import GridWorldEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from wrpsolver.bc.cunstomCnn import ResNet18
from wrpsolver.bc.eca_res import eca_resnet18
from wrpsolver.bc.timm import MobileNet
from sb3_contrib import RecurrentPPO

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int,  verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          self.model.save('rpp')
        return True
    
def make_env(env_id, rank = 0, logFile = None,seed=0):
    def _init():
        env = GridWorldEnv(channel=True)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v2'
    num_cpu = 8
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # callback = SaveOnBestTrainingRewardCallback(check_freq=10000)
    policy_kwargs = dict(
      features_extractor_class=ResNet18,
    )
    # model = RecurrentPPO("CnnLstmPolicy", env, verbose=1,batch_size=128,policy_kwargs=policy_kwargs)
    model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1,batch_size=2**10,n_steps=2**11,gamma=0.99,learning_rate=3e-4,ent_coef=0.01)
    model.learn(total_timesteps=2048*10000*4,progress_bar=True,log_interval=1)


