import shapely
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack
from wrpsolver.bc.gym_env_hwc_100_pp_dict import GridWorldEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from wrpsolver.bc.cunstomCnn import ResNet18
from wrpsolver.bc.eca_res import eca_resnet18
from stable_baselines3 import PPO

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int,  verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          i = self.n_calls / self.check_freq

          # testEnv = GridWorldEnv()
          # # testEnv = GridWorldEnv()
          # rewardList = []
          # for _ in range(5):
          #   observation = testEnv.reset()
          #   action,state = self.model.predict(observation,deterministic=False)
          #   Done = False
          #   rewardSum = 0
          #   while not Done:
          #     observation,reward,Done,_ = testEnv.step(int(action))
          #     action,state = self.model.predict(observation,state,deterministic=False)
          #     rewardSum += reward
          #   rewardList.append(rewardSum)
          # print(f"rewardList is {rewardList} after train {i} times")
          self.model.save('pp_time')
        return True
    
def make_env(env_id, rank = 0, logFile = None,seed=0):
    def _init():
        env = GridWorldEnv(channel=True)
        # env = MaxAndSkipEnv(env,4)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v2'
    num_cpu = 8
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000)
    policy_kwargs = dict(
      features_extractor_class=ResNet18,
    )
    # model = RecurrentPPO("MultiInputPolicy", env, verbose=1,batch_size=1024)
    # env = VecFrameStack(env,2)
    model = PPO("MultiInputPolicy", env, verbose=1,batch_size=2**10,n_steps=2**11,gamma=0.99,learning_rate=3e-4,ent_coef=0.01)
    # model.set_parameters("pp_rew")
    model.learn(total_timesteps=2048*10000*4,progress_bar=True,log_interval=1,callback=callback)


