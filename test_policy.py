import os
import json
import shapely
import cv2
from imitation.algorithms import bc
from random import choice
from wrpsolver.bc.gym_env import GridWorldEnv
from stable_baselines3.common.evaluation import evaluate_policy
pointList = [[21, 1], [21, 1], [21, 50], [33, 50], [34, 50], [34, 52], [33, 52], [1, 52], [1, 54], [1, 54], [1, 83], [1, 83], [1, 109], [1, 109], [1, 125], [28, 125], [28, 126], [28, 127], [28, 127], [1, 127], [1, 198], [65, 198], [65, 197], [65, 197], [65, 127], [43, 127], [42, 127], [42, 126], [43, 125], [68, 125], [68, 126], [68, 127], [68, 127], [67, 127], [67, 198], [110, 198], [110, 127], [82, 127], [82, 127], [82, 126], [82, 125], [92, 125], [92, 114], [92, 114], [92, 102], [92, 102], [94, 102], [94, 102], [94, 125], [139, 125], [139, 126], [139, 127], [139, 127], [111, 127], [111, 128], [111, 129], [111, 198], [184, 198], [184, 127], [154, 127], [153, 127], [153, 126], [154, 126], [184, 126], [184, 27], [94, 27], [94, 27], [94, 49], [94, 49], [94, 70], [94, 70], [92, 70], [92, 70], [92, 60], [92, 60], [92, 52], [78, 52], [78, 52], [78, 50], [78, 50], [91, 50], [91, 1]]
startPoint = (37, 50)

policy = bc.reconstruct_policy('/home/nianba/bc_policy.th')
dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
testJsonDir = dirPath + choice(picDirNames) + '/data.json'
with open(testJsonDir) as json_file:
    json_data = json.load(json_file)
polygon = shapely.Polygon(pointList)
polygon = shapely.Polygon(json_data['polygon'])
env = GridWorldEnv(polygon)
for i in range(10):
    observation = env.reset(startPoint)
    Done = False
    state = None
    action ,state= policy.predict(observation,state)
    action = int(action)
    while not Done:
        observation,reward,Done,_ = env.step(action)
        cv2.imshow('aa',observation)
        cv2.waitKey(0)
        action ,state = policy.predict(observation,state)
        action = int(action)
        print(action)
print('reward: ',reward)