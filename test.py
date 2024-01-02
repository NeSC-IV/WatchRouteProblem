
import time
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv,DrawPolygon
from stable_baselines3 import PPO
from nearest import NearestPolicy
from frontier import FrontierPolicy

if __name__ == "__main__":

    env = GridWorldEnv(render=False)
    test_times = 100

    TEST_RL = 0
    TEST_NE = 1
    TEST_FT = 0

    #for rl method
    if TEST_RL:
        lenList = []
        rewardList = []
        timeList = []
        model = PPO.load('saved_model/40_3_new')
        num = 0
        while num < test_times:

            observation,_ = env.reset()
            Done = False
            state = None
            action ,state= model.predict(observation,state,deterministic=True)
            rewardSum = 0
            cnt = 0
            while not Done:
                action = int(action)
                observation,reward,Done,_,_ = env.step(action)
                rewardSum += reward
                cnt += 1
                if not Done:
                    time_start = time.perf_counter()
                    action ,state = model.predict(observation,state,deterministic=True)
                    time_end = time.perf_counter()
                    timeList.append(time_end - time_start)
            if (rewardSum > 1):
                lenList.append(cnt)
                rewardList.append(rewardSum)
                print(rewardSum,cnt,num)
                num += 1
        print("rl method reward:",sum(rewardList)/test_times)
        print("rl method length:",sum(lenList)/test_times)
        print("rl method time:",sum(timeList)/sum(lenList))

    #for nearest method
    if TEST_NE:
        lenList = []
        rewardList = []
        timeList = []
        num = 0
        while num < test_times:
            observation,_ = env.reset()
            image = env.initImage.copy()
            image.fill(0)
            DrawPolygon(list((env.polygon.buffer(-1, join_style=2)).exterior.coords), 255, image, zoomRate = 1)
            Done = False
            state = None
            action = NearestPolicy(env.frontierList, env.pos, image, 3,env.polygon)
            if action==None:
                continue
            rewardSum = 0
            cnt = 1
            while not Done:
                action = int(action)
                observation,reward,Done,_,_ = env.step(action)
                cnt += 1
                rewardSum += reward
                if not Done:
                    time_start = time.perf_counter()
                    action = NearestPolicy(env.frontierList, env.pos, image, 3,env.polygon)
                    time_end = time.perf_counter()
                    timeList.append(time_end - time_start)
                    if(action == None):
                        break

            if (rewardSum > 1):
                lenList.append(cnt)
                rewardList.append(rewardSum)
                print(rewardSum,cnt,num)
                num += 1

        print("ne method reward:",sum(rewardList)/test_times)
        print("ne method length:",sum(lenList)/test_times)
        print("ne method time:",sum(timeList)/sum(lenList))

    #for nearest method
    if TEST_FT:
        lenList = []
        rewardList = []
        timeList = []
        num = 0
        while num < test_times:
            observation,_ = env.reset()
            image = env.initImage.copy()
            image.fill(0)
            DrawPolygon(list((env.polygon.buffer(-1, join_style=2)).exterior.coords), 255, image, zoomRate = 1)
            Done = False
            state = None
            actionList = FrontierPolicy(env.frontierList, env.pos, image, 3,env.polygon)
            if actionList==None:
                continue
            rewardSum = 0
            cnt = 1
            while not Done:
                for action in actionList:
                    action = int(action)
                    observation,reward,Done,_,_ = env.step(action)
                    cnt += 1
                    rewardSum += reward
                    if Done:
                        break

                if not Done:
                    time_start = time.perf_counter()
                    actionList = FrontierPolicy(env.frontierList, env.pos, image, 3,env.polygon)
                    time_end = time.perf_counter()
                    timeList.append(time_end - time_start)
                    if(actionList == None):
                        break

            if (rewardSum > 1):
                lenList.append(cnt)
                rewardList.append(rewardSum)
                print(rewardSum,cnt,num)
                num += 1

        print("ft method reward:",sum(rewardList)/test_times)
        print("ft method length:",sum(lenList)/test_times)
        print("ft method time:",sum(timeList)/sum(lenList))