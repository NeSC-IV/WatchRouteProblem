import os
import json
import shapely
import cv2
from imitation.algorithms import bc
from random import choice
from wrpsolver.bc.gym_env import GridWorldEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
polygon = [[1, 0], [0, 1], [0, 41], [37, 41], [37, 26], [37, 26], [38, 26], [38, 26], [38, 42], [44, 42], [45, 42], [45, 43], [44, 43], [41, 43], [41, 43], [38, 43], [38, 44], [38, 45], [38, 49], [38, 49], [38, 54], [39, 54], [39, 59], [39, 59], [39, 63], [39, 64], [39, 68], [39, 68], [39, 70], [39, 70], [39, 70], [80, 70], [80, 64], [80, 64], [80, 59], [80, 59], [80, 55], [80, 55], [81, 55], [81, 55], [81, 57], [81, 57], [81, 61], [81, 61], [81, 64], [81, 64], [81, 68], [81, 68], [81, 70], [81, 70], [81, 71], [81, 71], [81, 71], [81, 75], [81, 76], [81, 81], [81, 81], [81, 87], [81, 87], [81, 89], [100, 89], [100, 89], [100, 90], [100, 90], [81, 90], [81, 102], [81, 102], [81, 125], [81, 125], [81, 136], [84, 136], [84, 136], [84, 137], [84, 137], [81, 137], [81, 137], [80, 137], [80, 137], [80, 137], [64, 137], [64, 137], [64, 136], [64, 136], [80, 136], [80, 125], [80, 125], [80, 102], [80, 102], [80, 90], [80, 90], [80, 87], [80, 87], [80, 81], [80, 81], [80, 76], [80, 75], [80, 71], [38, 71], [38, 136], [57, 136], [58, 136], [58, 137], [57, 137], [39, 137], [39, 198], [122, 198], [122, 137], [91, 137], [91, 137], [91, 136], [91, 136], [122, 136], [122, 90], [107, 90], [107, 90], [107, 89], [107, 89], [122, 89], [122, 87], [122, 87], [122, 83], [122, 83], [122, 79], [122, 79], [122, 75], [121, 74], [121, 70], [121, 70], [121, 66], [121, 66], [121, 62], [121, 62], [121, 57], [121, 57], [121, 53], [121, 53], [121, 49], [121, 49], [121, 44], [121, 44], [121, 43], [110, 43], [110, 43], [99, 43], [99, 43], [99, 42], [99, 42], [109, 42], [109, 42], [135, 42], [135, 1], [123, 1], [123, 1], [101, 1], [101, 0], [91, 0], [91, 42], [92, 42], [92, 42], [92, 43], [92, 43], [90, 43], [90, 43], [88, 43], [88, 43], [88, 42], [88, 42], [90, 42], [90, 0], [38, 0], [38, 15], [38, 15], [37, 15], [37, 15], [37, 0]]
polygon = shapely.Polygon(polygon)
startPoint = (37,23)
# policy = bc.reconstruct_policy('bc_policy')
model = PPO.load('gail_policy')
# policy = model.policy.load('bc_policy') 
policy = model.policy
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
        action ,state = policy.predict(observation,state)
        print(action,reward)
        rewardSum += reward
    rewardList.append(rewardSum)
print('reward: ',rewardList)