from collections import defaultdict
import json
import os
import numpy as np
import pickle
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
dirPath = os.path.dirname(os.path.abspath(__file__))+'/wrpsolver/Test/pic_data_picsize100/'
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]


def countUnkown(image):
    ret,thresh=cv2.threshold(image,254,255,cv2.THRESH_BINARY_INV)
    cnt = cv2.countNonZero(thresh)
    return cnt

lock = Lock()
expert_trajs = defaultdict(list)
expert_lengths = []
expert_rewards = []

def get_expert_traj(picDataDir):

    filesNames = os.listdir(picDataDir)
    filesNames.remove('data.json')
    picIDs = [int(fileName.split('.')[0]) for fileName in filesNames]
    if(len(picIDs) < 10):
        return None
    picIDs.sort()
    picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in picIDs]
    pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(100,100,1) for picDir in picDirs]
    dataDir = picDataDir+'/data.json'
    with open(dataDir) as json_file:
        json_data = json.load(json_file)
    actionList = json_data['actionArray'][:len(pics)-1]

    traj = []
    for i in range(1,len(pics)):
        prev_obs = pics[i-1]
        action = actionList[i-1]
        obs = pics[i]
        
        # time.sleep(0.1)
        # cv2.imwrite('test/test.png',obs)
        stateUnknown = countUnkown(prev_obs[0])
        nextStateUnknown = countUnkown(obs[0])
        reward = (stateUnknown - nextStateUnknown) * 0.00005 * 2
        reward -= -0.0001
        reward *= 10
        reward = np.sign(reward) * np.log(1.0 + abs(reward))


        if(i == len(pics)-2):
            done = True
        else:
            done = False

        traj.append((np.array(prev_obs), np.array(action, dtype='int64'), reward, np.array(obs), done))
    return traj


def get_data_stats(d, rewards, lengths):

    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))

def main():

    # executor = ThreadPoolExecutor(max_workers=30)
    # for _ in executor.map(get_expert_traj,picDataDirs[:1]):
    #     pass
    trajectories = []
    cnt = 0
    for picDataDir in picDataDirs[:]:
        traj = get_expert_traj(picDataDir)
        if traj is not None:
            trajectories.append(traj)
            cnt += 1
            print(cnt)
    print("num episodes", len(trajectories))
    with open('expert_for_exploration_hwc_100.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    exit()

if __name__ == '__main__':
    main()
