import shapely
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from wrpsolver.bc.gym_env_hwc import GridWorldEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from wrpsolver.bc.cunstomCnn import ResNet18,ResNet34
from wrpsolver.bc.timm import EfficientnetB0,FbNetv3,MobileNet,MobileVit,XCIT,Tinynet,EfficientnetB4

import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

polygon = [[40, 1], [40, 1], [40, 61], [84, 61], [84, 60], [84, 60], [85, 60], [85, 60], [86, 60], [86, 60], [86, 61], [86, 61], [86, 61], [86, 70], [86, 70], [84, 70], [84, 70], [84, 62], [38, 62], [38, 63], [1, 63], [1, 63], [1, 149], [55, 149], [55, 120], [55, 120], [55, 91], [55, 91], [55, 90], [56, 89], [57, 89], [57, 89], [84, 89], [84, 81], [84, 81], [86, 81], [86, 81], [86, 91], [86, 91], [86, 91], [86, 91], [85, 91], [85, 109], [85, 109], [85, 127], [85, 127], [85, 134], [87, 134], [87, 134], [87, 136], [87, 136], [85, 136], [85, 164], [85, 164], [85, 196], [165, 196], [165, 136], [98, 136], [98, 136], [98, 134], [98, 134], [165, 134], [165, 134], [165, 129], [166, 129], [166, 122], [166, 122], [166, 115], [166, 115], [166, 108], [166, 108], [166, 101], [166, 101], [166, 94], [166, 94], [166, 87], [166, 87], [166, 80], [167, 80], [167, 73], [167, 73], [167, 66], [167, 66], [167, 59], [167, 59], [167, 57], [125, 57], [125, 57], [125, 55], [125, 55], [169, 55], [169, 55], [199, 55], [199, 1], [115, 1], [115, 48], [115, 48], [114, 48], [114, 48], [114, 1], [86, 1], [86, 14], [86, 14], [86, 37], [86, 37], [86, 49], [86, 49], [84, 49], [84, 49], [84, 41], [84, 41], [84, 1]]
polygon = shapely.Polygon(polygon)
startPoint = (55, 65)
class SaveOnBestTrainingRewardCallback(BaseCallback):

    def __init__(self, check_freq: int,  verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          i = self.n_calls / self.check_freq

          testEnv = GridWorldEnv(polygon,startPoint,)
          # testEnv = GridWorldEnv()
          rewardList = []
          for _ in range(2):
            observation = testEnv.reset()
            action,state = self.model.predict(observation,deterministic=True)
            Done = False
            rewardSum = 0
            while not Done:
              observation,reward,Done,_ = testEnv.step(int(action))
              action,state = self.model.predict(observation,state,deterministic=False)
              rewardSum += reward
            rewardList.append(rewardSum)
          print(f"rewardList is {rewardList} after train {i} times")
          self.model.save('ppo_res18_expo')
        return True
    
def make_env(env_id, rank, logFile = None,seed=0):
    def _init():
        # env = GridWorldEnv(polygon,startPoint)
        env = GridWorldEnv()
        return Monitor(env)
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v0'
    num_cpu = 8  # Number of processes to use
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # env = VecNormalize(env, norm_obs=False, norm_reward=True,clip_obs=10.)
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000)
    policy_kwargs = dict(
      features_extractor_class=ResNet18,
    )
    model =PPO("CnnPolicy",env,verbose=1,n_steps=512,gamma=0.99,batch_size=1024,policy_kwargs=policy_kwargs)
    model.set_parameters('ppo_res18_expo')
    model.learn(total_timesteps=2048*10000*4,progress_bar=True,log_interval=10,callback=callback)


