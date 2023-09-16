from collections import defaultdict
import json
import os
import pickle
import shapely
from multiprocessing import Pool,Lock,Value,Manager
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv
DIRPATH = os.path.dirname(os.path.abspath(__file__))+'/wrpsolver/Test/optimal_path/'
JSONPATHS = os.listdir(DIRPATH)
ACTIONDICT =    {
                    (1,0):0,(-1,0):1,(0,1):2,(0,-1):3,
                }

# import random
# random.shuffle ( JSONPATHS )

lock = Lock()
length = Value('i', 0)
manager = Manager()
trajectories = manager.list()
def GetSingleTrajectory(jsonName):
    env = GridWorldEnv(render=False)
    with open(DIRPATH+jsonName) as f:
        jsonData = json.load(f)
    polygon = shapely.Polygon(jsonData['polygon'])
    paths = jsonData["paths"]
    startPoint = paths[0][0]
    actionList = Path2Action(paths)

    traj = []
    obs,_ = env.reset(polygon=polygon,startPoint=startPoint)
    Done = False
    rewardSum = 0
    cnt = 0
    for action in actionList:
        next_obs,reward,Done,_,_ = env.step(action)
        traj.append((obs, action, reward, next_obs, Done))
        rewardSum += reward
        cnt += 1
        obs = next_obs
        if Done:
            break
    lock.acquire()
    if (rewardSum > 1.5) and (cnt > 70):
            trajectories.append(traj)
            length.value += 1
            print(rewardSum,cnt,length.value)
    else:
        print(rewardSum,cnt)
    lock.release()
    return length.value

def Path2Action(paths):
    path = []
    actionList = []
    for p in paths:
        path += p[:-1]
    for i in range(len(path) - 1):
        pos = path[i]
        posNext = path[i+1]
        action = ACTIONDICT[posNext[0]-pos[0],posNext[1]-pos[1]]
        actionList.append(action)
    return actionList

class getTrajectory():
    def __init__(self) -> None:
        self.pool = Pool(16)

    def CheckTerminate(self, arg):
        global trajectories
        if arg >= 10000:
            self.pool.terminate()
            print("num episodes", len(trajectories))
            trajectories = list(trajectories)
            with open('demonstrations_expo.pkl', 'wb') as f:
                pickle.dump(trajectories, f)

    def SaveTrajectory(self):
        for p in JSONPATHS:
            self.pool.apply_async(func=GetSingleTrajectory,args=(p,),callback=self.CheckTerminate)
        self.pool.close()
        self.pool.join()

def main():

    trajectories = []
    length = 0
    env = GridWorldEnv(render=True)
    for jsonName in JSONPATHS[29:30]:
        with open(DIRPATH+jsonName) as f:
            jsonData = json.load(f)
        polygon = shapely.Polygon(jsonData['polygon'])
        paths = jsonData["paths"]
        startPoint = paths[0][0]
        actionList = Path2Action(paths)

        traj = []
        obs,_ = env.reset(polygon=polygon,startPoint=startPoint)
        Done = False
        rewardSum = 0
        cnt = 0
        for action in actionList:
            next_obs,reward,Done,_,_ = env.step(action)
            traj.append((obs, action, reward, next_obs, Done))
            rewardSum += reward
            cnt += 1
            obs = next_obs
            if Done:
                break
        if (rewardSum > 1.5) and (cnt > 70):
            trajectories.append(traj)
            length += 1
            print(rewardSum,cnt,length)
        else:
            print(rewardSum,cnt)
        if (length >= 1000):
            break
    print("num episodes", len(trajectories))
    with open('demonstrations_expo.pkl', 'wb') as f:
        pickle.dump(trajectories, f)



if __name__ == '__main__':
    if True:
        gt = getTrajectory()
        gt.SaveTrajectory()
    else:
        main()