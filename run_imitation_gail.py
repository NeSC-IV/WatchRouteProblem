import os
import cv2
import json
import numpy as np
import shutil
import shapely
import torch as th
from stable_baselines3 import PPO
from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import CnnRewardNet
from random import choice,shuffle
from multiprocessing import Pool,Manager
from wrpsolver.bc.gym_env_hwc import GridWorldEnv
from wrpsolver.bc.cunstomCnn import ResNet18

polygon = [[40, 1], [40, 1], [40, 61], [84, 61], [84, 60], [84, 60], [85, 60], [85, 60], [86, 60], [86, 60], [86, 61], [86, 61], [86, 61], [86, 70], [86, 70], [84, 70], [84, 70], [84, 62], [38, 62], [38, 63], [1, 63], [1, 63], [1, 149], [55, 149], [55, 120], [55, 120], [55, 91], [55, 91], [55, 90], [56, 89], [57, 89], [57, 89], [84, 89], [84, 81], [84, 81], [86, 81], [86, 81], [86, 91], [86, 91], [86, 91], [86, 91], [85, 91], [85, 109], [85, 109], [85, 127], [85, 127], [85, 134], [87, 134], [87, 134], [87, 136], [87, 136], [85, 136], [85, 164], [85, 164], [85, 196], [165, 196], [165, 136], [98, 136], [98, 136], [98, 134], [98, 134], [165, 134], [165, 134], [165, 129], [166, 129], [166, 122], [166, 122], [166, 115], [166, 115], [166, 108], [166, 108], [166, 101], [166, 101], [166, 94], [166, 94], [166, 87], [166, 87], [166, 80], [167, 80], [167, 73], [167, 73], [167, 66], [167, 66], [167, 59], [167, 59], [167, 57], [125, 57], [125, 57], [125, 55], [125, 55], [169, 55], [169, 55], [199, 55], [199, 1], [115, 1], [115, 48], [115, 48], [114, 48], [114, 48], [114, 1], [86, 1], [86, 14], [86, 14], [86, 37], [86, 37], [86, 49], [86, 49], [84, 49], [84, 49], [84, 41], [84, 41], [84, 1]]
polygon = shapely.Polygon(polygon)
startPoint = (55, 65)

def getTrajectories(args):
    picDataDir = args[0]
    # picDataDir = '/remote-home/ums_qipeng/WatchRouteProblem/wrpsolver/Test/pic_data/pic_data/0952dd0899830bfd0006b12863318943'
    trajectories = args[1]
    filesNames = os.listdir(picDataDir)
    if(len(filesNames) < 30):
        return
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

if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    dirPath = os.path.dirname(os.path.abspath(__file__))+"/wrpsolver/Test/pic_data/"
    picDirNames = os.listdir(dirPath)
    picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
    shuffle(picDataDirs)
    cnt = 0
    rewardList = []
    trajectories = Manager().list()
    rng = np.random.default_rng(0)
    venv = make_vec_env('IL/GridWorld-v0', n_envs=4, rng=rng,parallel=True)
    # venv = GridWorldEnv(polygon,startPoint)
    policy_kwargs = dict(
      features_extractor_class=ResNet18,
    )
    learner =PPO("CnnPolicy",venv,verbose=1,n_steps=512,gamma=0.99,batch_size=1024,policy_kwargs=policy_kwargs)
    learner.set_parameters('gail_policy')
    reward_net = CnnRewardNet(
        venv.observation_space,
        venv.action_space,
        hwc_format=False,
    ).to(device)

    pool = Pool(48)
    pool.map(getTrajectories,iterable = [(picDataDir,trajectories) for picDataDir in picDataDirs])
    pool.close()
    pool.join()
# try:
    transitions = flatten_trajectories(trajectories)
    trajectories = []
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=128,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon = True
    )
    # gail_trainer.train(16384*1000)
    gail_trainer.train(16384*1000)
# except Exception as e:
#     print(e)
# finally:
    learner.save('gail_policy')
#     pass