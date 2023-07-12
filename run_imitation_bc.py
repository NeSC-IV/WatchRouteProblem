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
from stable_baselines3 import PPO
from typing import Callable
from wrpsolver.bc.gym_env_hwc import GridWorldEnv
from wrpsolver.bc.cunstomCnn import ResNet18,ResNet34
from wrpsolver.bc.timm import EfficientnetB0,FbNetv3,MobileNet,MobileVit,XCIT,Tinynet,EfficientnetB4

dirPath = os.path.dirname(os.path.abspath(__file__))+"/wrpsolver/Test/pic_data/"
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
picDataDirs.sort()
trajectories = Manager().list()
env = GridWorldEnv()
rng = np.random.default_rng(0)

def getTrajectories(args):
    picDataDir = args[0]
    trajectories = args[1]
    filesNames = os.listdir(picDataDir)
    if(len(filesNames) < 30):
        return
    # picDataDir = '/remote-home/ums_qipeng/WatchRouteProblem/wrpsolver/Test/pic_data/pic_data/0952dd0899830bfd0006b12863318943'
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
    else:
        trajectories.append(trajectory)
    # print(len(trajectories))

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        # return max((progress_remaining**3) * initial_value,1e-8)
        return 3e-5

    return func

pool = Pool(24)
pool.map(getTrajectories,iterable = [(picDataDir,trajectories) for picDataDir in picDataDirs])
pool.close()
pool.join()
policy_kwargs = dict(
    features_extractor_class=ResNet18,
)
# model = PPO("CnnPolicy", env, verbose=1,n_steps=512,gamma=0.999,batch_size=2048,policy_kwargs=policy_kwargs)
model = PPO("CnnPolicy", env, verbose=1,n_steps=512,gamma=0.999,batch_size=2048)
model.set_parameters('bc_policy_res18.zip')
transitions = flatten_trajectories(trajectories)
trajectories = []
bc_trainer = bc.BC(
    observation_space = env.observation_space,
    action_space = env.action_space,
    demonstrations=transitions,
    rng=rng,
    policy=model.policy,
    batch_size=2**12
)
bc_trainer.train(n_epochs=1000,log_interval=10)
# bc_trainer.policy.save('bc_policy_single')
model.save('bc_policy_res18.zip')