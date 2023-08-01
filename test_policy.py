import os
import json
import cv2
from random import choice
from wrpsolver.bc.gym_env_hwc_100_pp_dict import GridWorldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import SubprocVecEnv,VecFrameStack
from stable_baselines3.common.monitor import Monitor
if __name__ == "__main__":
    dirPath = os.path.dirname(os.path.abspath(__file__))+"/wrpsolver/Test/pic_data_picsize100_pos/"
    picDirNames = os.listdir(dirPath)
    testJsonDir = dirPath + choice(picDirNames) + '/data.json'
    with open(testJsonDir) as json_file:
        json_data = json.load(json_file)

    # def make_env(env_id, rank = 0, logFile = None,seed=0):
    #     def _init():
    #         env = GridWorldEnv(channel=False,render=True)
    #         env = MaxAndSkipEnv(env,4)
    #         return Monitor(env)
    #     return _init
    # env = SubprocVecEnv([make_env(0, i) for i in range(1)])
    # env = VecFrameStack(env,2)
    env = GridWorldEnv(channel=True,render=True)
    model = PPO.load('pp_rew_gamma')
    rewardList = []
    for i in range(1):
        observation,_ = env.reset()
        Done = False
        state = None
        action ,state= model.predict(observation,state,deterministic=True)
        rewardSum = 0
        cnt = 0
        while not Done:
            action = int(action)
            observation,reward,Done,_,_ = env.step(action)
            # cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp/'+str(cnt)+'.png',env.globalObs)
            action ,state = model.predict(observation,state,deterministic=True)
            print(action,reward,cnt)
            rewardSum += reward
            cnt += 1
        rewardList.append(rewardSum)
    print('reward: ',rewardList)