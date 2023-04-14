import os
import cv2
import json
import numpy as np
import shutil
from stable_baselines3 import A2C
from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms.adversarial.airl import AIRL
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import CnnRewardNet
from stable_baselines3.ppo import CnnPolicy
from random import choice,shuffle
from multiprocessing import Pool,Manager
import wrpsolver.bc.gym_env

dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
shuffle(picDataDirs)
cnt = 0
rewardList = []
trajectories = Manager().list()
rng = np.random.default_rng(0)
venv = make_vec_env('IL/GridWorld-v0', n_envs=8, rng=rng)
# venv = env
learner = A2C(policy=CnnPolicy,env = venv)
# learner.policy = learner.policy.load('bc_policy_ppo1')
reward_net = CnnRewardNet(
    venv.observation_space,
    venv.action_space,
)

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

pool = Pool(32)
pool.map(getTrajectories,iterable = [(picDataDir,trajectories) for picDataDir in picDataDirs[:1000]])
pool.close()
pool.join()

try:
    transitions = flatten_trajectories(trajectories)
    trajectories = []
    gail_trainer = AIRL(
        demonstrations=transitions,
        demo_batch_size=512,
        gen_replay_buffer_capacity=256,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon = True
    )
    gail_trainer.train(16384*10)
except Exception as e:
    print(e)
finally:
    learner.save('airl_policy')
    pass

print(rewardList)
