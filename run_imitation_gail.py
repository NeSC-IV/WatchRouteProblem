import os
import cv2
import json
import numpy as np
import shutil
import shapely
import gym
from stable_baselines3 import PPO
from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.ppo import CnnPolicy
from random import choice,shuffle
from stable_baselines3.common.env_checker import check_env

from wrpsolver.bc.gym_env import GridWorldEnv

dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
shuffle(picDataDirs)
cnt = 0
rewardList = []
trajectories = []
pointList = [[21, 1], [21, 1], [21, 50], [33, 50], [34, 50], [34, 52], [33, 52], [1, 52], [1, 54], [1, 54], [1, 83], [1, 83], [1, 109], [1, 109], [1, 125], [28, 125], [28, 126], [28, 127], [28, 127], [1, 127], [1, 198], [65, 198], [65, 197], [65, 197], [65, 127], [43, 127], [42, 127], [42, 126], [43, 125], [68, 125], [68, 126], [68, 127], [68, 127], [67, 127], [67, 198], [110, 198], [110, 127], [82, 127], [82, 127], [82, 126], [82, 125], [92, 125], [92, 114], [92, 114], [92, 102], [92, 102], [94, 102], [94, 102], [94, 125], [139, 125], [139, 126], [139, 127], [139, 127], [111, 127], [111, 128], [111, 129], [111, 198], [184, 198], [184, 127], [154, 127], [153, 127], [153, 126], [154, 126], [184, 126], [184, 27], [94, 27], [94, 27], [94, 49], [94, 49], [94, 70], [94, 70], [92, 70], [92, 70], [92, 60], [92, 60], [92, 52], [78, 52], [78, 52], [78, 50], [78, 50], [91, 50], [91, 1]]
polygon = shapely.Polygon(pointList)
rng = np.random.default_rng(0)
# env = GridWorldEnv(polygon)
# env = gym.vector.make('IL/GridWorld-v0',8)
venv = make_vec_env('IL/GridWorld-v0', n_envs=1, rng=rng)
# venv = env
learner = PPO(env=venv, policy=CnnPolicy)
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=None,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)
try:
    for picDataDir in picDataDirs:
        print('第 ' + str(cnt) +' 次训练开始：')
        filesNames = os.listdir(picDataDir)
        filesNames.remove('data.json')
        picIDs = [int(fileName.split('.')[0]) for fileName in filesNames]
        picIDs.sort()
        picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in picIDs]
        dataDir = picDataDir+'/data.json'
        pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(200,200,1) for picDir in picDirs]
        with open(dataDir) as json_file:
            json_data = json.load(json_file)
        actionList = json_data['actionArray'][:len(pics)-1]
        pics = pics[:len(actionList)+1]

        cnt += 1
        try:
            trajectory = Trajectory(pics,actionList,None,True)
        except Exception as e:
            print(e)
            print(picDataDir)
            shutil.rmtree(picDataDir)
            exit(0)
            
        trajectories.append(trajectory)
        if(cnt % 200 == 0):
            transitions = flatten_trajectories(trajectories)
            gail_trainer.set_demonstrations(transitions)
            gail_trainer.train(20000)
            trajectories = []
finally:
    learner.save('/home/nianba/gail_policy.pk1')

print(rewardList)