import shapely
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from wrpsolver.bc.gym_env_hwc_100_pp_dict import GridWorldEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from wrpsolver.bc.cunstomCnn import ResNet18

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int,  verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        # if self.n_calls % self.check_freq == 0:
          # i = self.n_calls / self.check_freq

          # testEnv = GridWorldEnv(polygon,startPoint,)
          # # testEnv = GridWorldEnv()
          # rewardList = []
          # for _ in range(1):
          #   observation = testEnv.reset()
          #   action,state = self.model.predict(observation,deterministic=True)
          #   Done = False
          #   rewardSum = 0
          #   while not Done:
          #     observation,reward,Done,_ = testEnv.step(int(action))
          #     action,state = self.model.predict(observation,state,deterministic=False)
          #     rewardSum += reward
          #   rewardList.append(rewardSum)
          # print(f"rewardList is {rewardList} after train {i} times")
        #   self.model.save('dqn_pp_gamma')
        return True
    
def make_env(env_id, rank, logFile = None,seed=0):
    def _init():
        # env = GridWorldEnv(polygon,startPoint)
        env = GridWorldEnv()
        return Monitor(env)
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v0'
    num_cpu = 4  # Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = VecNormalize(env, norm_obs=False, norm_reward=True,clip_obs=10.)
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000)
    policy_kwargs = dict(
      features_extractor_class=ResNet18,
    )
    model =DQN("MultiInputPolicy",env,verbose=1,gamma=0.99,batch_size=1024,exploration_initial_eps = 1,exploration_final_eps = 0.2)
    model.learn(total_timesteps=5e7,progress_bar=True,log_interval=1,callback=callback)


