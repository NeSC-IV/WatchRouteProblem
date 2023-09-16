from collections import defaultdict
import numpy as np
import pickle
from wrpsolver.bc.gym_env_hwc_100_pp_dict import GridWorldEnv
from stable_baselines3 import PPO

expert_trajs = defaultdict(list)
def GetFlattenObs(obs):
    v = []
    agent = obs['agent']
    localImage = obs['localImage']
    localImage1 = obs['localImage1']
    v = agent.tolist()
    localImageList = localImage.reshape(3*3).tolist()
    localImage1List = np.array(localImage1.reshape(130*130),dtype=np.int64).tolist()
    v = np.array(v + localImageList + localImage1List)
    return v
def main():

    trajectories = []
    length = 0
    model = PPO.load('pp_res')
    env = GridWorldEnv(render=False)
    while True:
        traj = []
        obs,_ = env.reset()
        Done = False
        rewardSum = 0
        cnt = 0
        while not Done:
            action , _= model.predict(obs,None,deterministic=True)
            action = int(action)
            next_obs,reward,Done,_,_ = env.step(action)
            traj.append((obs, action, reward, next_obs, Done))
            rewardSum += reward
            cnt += 1
            obs = next_obs
        if (rewardSum > 2) and (cnt > 50):
            trajectories.append(traj)
            length += 1
            print(rewardSum,cnt,length)
        else:
            print(rewardSum,cnt)
        if (length >= 10000):
            break
    print("num episodes", len(trajectories))
    with open('demonstrations.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    exit()

if __name__ == '__main__':

    main()