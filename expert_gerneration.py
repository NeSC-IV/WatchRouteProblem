from collections import defaultdict
import json
import os
import numpy as np
import pickle
import cv2
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
dirPath = os.path.dirname(os.path.abspath(__file__))+'/wrpsolver/Test/pic_data/pic_data/'
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
    global expert_trajs
    # picDataDir= dirPath + '0a6b669e61969b02444bbe46cac90bb6'
    # picDataDir = '/remote-home/ums_qipeng/WatchRouteProblem/wrpsolver/Test/pic_data/pic_data/0952dd0899830bfd0006b12863318943'
    filesNames = os.listdir(picDataDir)
    filesNames.remove('data.json')
    picIDs = [int(fileName.split('.')[0]) for fileName in filesNames]
    if(len(picIDs) < 50):
        return
    picIDs.sort()
    picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in picIDs]
    pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE).reshape(1,200,200) for picDir in picDirs]
    dataDir = picDataDir+'/data.json'
    with open(dataDir) as json_file:
        json_data = json.load(json_file)
    actionList = json_data['actionArray'][:len(pics)-1]

    episode_reward = 0
    traj = []

    episode_infos = None
    for i in range(len(pics)-1):
        state = pics[i]
        next_state = pics[i+1]
        action = actionList[i]
        stateUnknown = countUnkown(state[0])
        nextStateUnknown = countUnkown(next_state[0])
        reward = stateUnknown - nextStateUnknown
        reward *=  0.00003

        if(i == len(pics)-2):
            done = True
        else:
            done = False

        traj.append((state, next_state, action, reward, done))
        episode_reward += (reward-0.0001)



    if episode_reward > 0:
        lock.acquire()
        states, next_states, actions, rewards, dones = zip(*traj)

        expert_trajs["states"].append(states)
        expert_trajs["next_states"].append(next_states)
        expert_trajs["actions"].append(actions)
        expert_trajs["rewards"].append(rewards)
        expert_trajs["dones"].append(dones)
        expert_trajs["lengths"].append(len(traj))
        expert_lengths.append(len(traj))
        expert_rewards.append(episode_reward)
        print('Ep {}\tSaving Episode reward: {:.2f}\t'.format(len(expert_trajs["states"]), episode_reward))
        lock.release()

# for k, v in expert_trajs.items():
#     expert_trajs[k] = np.array(v)

def get_data_stats(d, rewards, lengths):
    # lengths = d["lengths"]

    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))

def main():

    executor = ThreadPoolExecutor(max_workers=30)
    for _ in executor.map(get_expert_traj,picDataDirs[:200]):
        pass
    get_data_stats(expert_trajs, np.array(expert_rewards), np.array(expert_lengths))
    print('Final size of Replay Buffer: {}'.format(sum(expert_trajs["lengths"])))
    with open('expert_for_IQ_single.pkl', 'wb') as f:
        pickle.dump(expert_trajs, f)
    exit()

if __name__ == '__main__':
    main()
