import os
import json
import shapely
import cv2
from imitation.algorithms import bc
from random import choice
from wrpsolver.bc.gym_env import GridWorldEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

# policy = bc.reconstruct_policy('/home/nianba/bc_policy.th')
model = DQN.load('gail_policy')
dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
testJsonDir = dirPath + choice(picDirNames) + '/data.json'
with open(testJsonDir) as json_file:
    json_data = json.load(json_file)
env = GridWorldEnv()
rewardList = []
for i in range(10):
    observation = env.reset()
    Done = False
    state = None
    action ,state= model.predict(observation,state)
    action = int(action)
    rewardSum = 0
    while not Done:
        observation,reward,Done,_ = env.step(int(action))
        # cv2.imshow('aa',observation)
        # cv2.waitKey(0)
        action ,state = model.predict(observation,state)
        print(action)
        rewardSum += reward
    rewardList.append(reward)
print('reward: ',rewardList)