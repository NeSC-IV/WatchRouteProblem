import os
import cv2
import json
import numpy as np
import shutil
import torch as th
from multiprocessing import Pool,Manager
from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms import bc
from random import choice,shuffle
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.ppo.policies import CnnPolicy

from wrpsolver.bc.gym_env import GridWorldEnv

dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
shuffle(picDataDirs)
rewardList = []
trajectories = Manager().list()
env = GridWorldEnv()
rng = np.random.default_rng(0)

def getTrajectories(args):
    picDataDir = args[0]
    trajectories = args[1]
    filesNames = os.listdir(picDataDir)
    filesNames.remove('data.json')
    picIDs = [int(fileName.split('.')[0]) for fileName in filesNames]
    picIDs.sort()
    picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in picIDs]
    dataDir = picDataDir+'/data.json'
    pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(1,200,200) for picDir in picDirs]
    with open(dataDir) as json_file:
        json_data = json.load(json_file)
    actionList = json_data['actionArray'][:len(pics)-1]
    pics = pics[:len(actionList)+1]

    try:
        trajectory = Trajectory(pics,actionList,None,True)
    except Exception as e:
        print(e)
        print(picDataDir)
        shutil.rmtree(picDataDir)        
    trajectories.append(trajectory)
    print(len(trajectories))
pool = Pool(24)
pool.map(getTrajectories,iterable = [(picDataDir,trajectories) for picDataDir in picDataDirs])
pool.close()
pool.join()
# policy = CnnPolicy(env.observation_space,env.action_space,lambda _: 0.0003,)
policy = bc.reconstruct_policy('/home/nianba/bc_policy_ppo1.th')
transitions = flatten_trajectories(trajectories)
bc_trainer = bc.BC(
    observation_space = env.observation_space,
    action_space = env.action_space,
    demonstrations=transitions,
    rng=rng,
    batch_size=4096,
    policy= policy
)
bc_trainer.train(n_epochs=10,log_interval=100)
bc_trainer.save_policy(policy_path='/home/nianba/bc_policy_ppo1.th')