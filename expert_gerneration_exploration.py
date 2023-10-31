
import json
import os
import pickle
import shapely
from multiprocessing import Pool,Lock,Value,Manager
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv
DIRPATH = os.path.dirname(os.path.abspath(__file__))+'/wrpsolver/Test/optimal_path_60_3/'
JSONPATHS = os.listdir(DIRPATH)[:]
step = 3
ACTIONDICT =    {
                    (step,0):0,(-step,0):1,(0,step):2,(0,-step):3,
                    # (1,1):4,(-1,-1):5,(-1,1):6,(1,-1):7
                }
render = False
import random
random.shuffle ( JSONPATHS )

lock = Lock()
length = Value('i', 0)
manager = Manager()
trajectories = manager.list()
def GetSingleTrajectory(jsonName):
    env = GridWorldEnv(render=render)
    with open(DIRPATH+jsonName) as f:
        jsonData = json.load(f)
    polygon = shapely.Polygon(jsonData['polygon'])
    if(polygon.area > 30000 or polygon.area < 8000):
        return length.value
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
    if (rewardSum > 10) and (cnt > 20) and Done:
        trajectories.append(traj)
        length.value += 1
        print(rewardSum,cnt,length.value)
    else:
        print(rewardSum,cnt)
        # os.remove(DIRPATH+jsonName)
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
        self.pool = Pool(32)

    def CheckTerminate(self, arg):
        
        if arg >= 10000:
            self.pool.terminate()

    def SaveTrajectory(self):
        global trajectories
        for p in JSONPATHS[:]:
            self.pool.apply_async(func=GetSingleTrajectory,args=(p,),callback=self.CheckTerminate)
        self.pool.close()
        self.pool.join()
        print("num episodes", len(trajectories))
        trajectories = list(trajectories)
        with open('demonstrations_60_3.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

def main():

    trajectories = []
    length = 0
    env = GridWorldEnv(render=render)
    for jsonName in JSONPATHS[100:]:
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
        if (rewardSum > 1) and (cnt > 20):
            trajectories.append(traj)
            length += 1
            print(rewardSum,cnt,length)
        else:
            print(rewardSum,cnt)
        
        if (length >= 10000):
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