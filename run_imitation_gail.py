import os
import cv2
import json
import numpy as np
import shutil
import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import CnnRewardNet
from stable_baselines3.a2c import CnnPolicy
from random import choice,shuffle
import torch as th
from multiprocessing import Pool,Manager
import wrpsolver.bc.gym_env

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

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





if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
    picDirNames = os.listdir(dirPath)
    picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
    shuffle(picDataDirs)
    cnt = 0
    rewardList = []
    trajectories = Manager().list()
    rng = np.random.default_rng(0)
    venv = make_vec_env('IL/GridWorld-v0', n_envs=8, rng=rng,parallel=True)
    # venv = make_vec_env('IL/GridWorld-v0', n_envs=8, rng=rng)
    # venv = SubprocVecEnv([make_env('IL/GridWorld-v0', i) for i in range(16)])
    learner = PPO(env=venv, policy='CnnPolicy',batch_size=1024,n_steps=128)
    reward_net = CnnRewardNet(
        venv.observation_space,
        venv.action_space,
        hwc_format=False,
    ).to(device)



    pool = Pool(48)
    pool.map(getTrajectories,iterable = [(picDataDir,trajectories) for picDataDir in picDataDirs])
    pool.close()
    pool.join()
    try:
        transitions = flatten_trajectories(trajectories)
        trajectories = []
        gail_trainer = GAIL(
            demonstrations=transitions,
            demo_batch_size=512,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon = True
        )
        gail_trainer.train(16384*100)
    except Exception as e:
        print(e)
    finally:
        learner.save('gail_policy')
        pass