import os
import cv2
import json
import numpy as np
import shutil
from imitation.data import types
from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from torch.utils import data as th_data
from random import choice,shuffle

from wrpsolver.bc.gym_env import GridWorldEnv

dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
shuffle(picDataDirs)
cnt = 0
rewardList = []
trajectories = []
env = GridWorldEnv()
rng = np.random.default_rng(0)
policy = bc.reconstruct_policy('/home/nianba/bc_policy.th')
bc_trainer = bc.BC(
    observation_space = env.observation_space,
    action_space = env.action_space,
    demonstrations=None,
    rng=rng,
    policy= policy,
)
for picDataDir in picDataDirs:
    print('第 ' + str(cnt) +' 次训练开始：')
    filesNames = os.listdir(picDataDir)
    filesNames.remove('data.json')
    picIDs = [int(fileName.split('.')[0]) for fileName in filesNames]
    picIDs.sort()
    picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in picIDs]
    dataDir = picDataDir+'/data.json'
    pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE) for picDir in picDirs]
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
    if(cnt % 100 == 0):
        transitions = flatten_trajectories(trajectories)
        bc_trainer.set_demonstrations(transitions)
        bc_trainer.train(n_epochs=1)
        trajectories = []

bc_trainer.save_policy(policy_path='/home/nianba/bc_policy.th')
    
print(rewardList)