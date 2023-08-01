from collections import defaultdict
import json
import os
import numpy as np
import pickle
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
dirPath = os.path.dirname(os.path.abspath(__file__))+'/wrpsolver/Test/pic_data_picsize100_pos/'
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
    filesNames.remove('polygon.png')
    picIDs = [int(fileName.split('.')[0].split('_')[0]) for fileName in filesNames]
    maxpicID = max(picIDs)
    if(maxpicID < 10):
        return None
    picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in range(maxpicID+1)]
    picDirsPos = [picDataDir+'/'+str(picID)+'_pos.png' for picID in range(maxpicID+1)]
    picDirsft = [picDataDir+'/'+str(picID)+'_ft.png' for picID in range(maxpicID+1)]
    pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(100,100,1) for picDir in picDirs]
    picsPos = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(100,100,1) for picDir in picDirsPos]
    picsFt = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(100,100,1) for picDir in picDirsft]
    dataDir = picDataDir+'/data.json'
    with open(dataDir) as json_file:
        json_data = json.load(json_file)
    actionList = json_data['actionArray'][:len(pics)-1]

    traj = []
    totalReward = 0
    for i in range(1,len(pics)):
        prev_image = pics[i-1]
        action = actionList[i-1]
        image = pics[i]
        
        # time.sleep(0.1)
        # cv2.imwrite('test/test.png',obs)
        stateUnknown = countUnkown(prev_image)
        nextStateUnknown = countUnkown(image)
        reward = (stateUnknown - nextStateUnknown) * 0.0002
        reward = max(reward,0)
        reward = min(reward,0.5)
        reward -= 0.001
        # reward = np.sign(reward) * np.log(1.0 + abs(reward))
        totalReward += reward


        if(i == len(pics)-2):
            done = True
        else:
            done = False
        prev_obs = cv2.merge([pics[i-1],picsPos[i-1],picsFt[i-1]])
        obs = cv2.merge([pics[i],picsPos[i],picsFt[i]])
        traj.append((np.array(prev_obs), np.array(action, dtype='int64'), reward, np.array(obs), done))
    print(totalReward)
    return traj


def get_data_stats(d, rewards, lengths):

    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))

def main():

    trajectories = []
    cnt = 0
    for picDataDir in picDataDirs[:100]:
        traj = get_expert_traj(picDataDir)
        if traj is not None:
            trajectories.append(traj)
            cnt += 1
            print(cnt)
    print("num episodes", len(trajectories))
    with open('expert_for_exploration_hwc_1_pos.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    exit()

if __name__ == '__main__':
    main()
