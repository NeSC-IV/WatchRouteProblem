import os
import cv2
import sys
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv
from wrpsolver.Test.draw_pictures import DrawMultiline,DrawPath,DrawPolygon
from stable_baselines3 import PPO
path = "/remote-home/ums_qipeng/WatchRouteProblem/render_saved/"

def DrawFullPath(env,seed):
    image = env.initImage.copy()
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    image.fill(0)
    DrawPolygon(env.polygon, (255,255,255), image, 1, backgroundColor=(0))
    # cv2.imwrite(path+"fullPath/"+'polygon.png',image)
    DrawPath(image,env.path,color=(255,245,0))
    cv2.imwrite(path+'fullPath/'+seed+".png",image)

def MakeGif(folder,output,cnt):
    frames = []
    for pic in range(cnt):
        im = Image.open(path + folder + "/" + str(pic)+".png")
        frames.append(im)
    frame_one = frames[0]
    frame_one.save(path+output, format="GIF", append_images=frames,
            save_all=True, duration=60, loop=0)

def LogFolderInit():
    shutil.rmtree(path + "tmp0/",ignore_errors=True)
    shutil.rmtree(path + "tmp1/",ignore_errors=True)
    shutil.rmtree(path + "tmp2/",ignore_errors=True)
    os.makedirs(path + "tmp0/",exist_ok=True)
    os.makedirs(path + "tmp1/",exist_ok=True)
    os.makedirs(path + "tmp2/",exist_ok=True)
    os.makedirs(path + "range_pathLength/",exist_ok=True)


def PlotRangeLength(range,length,fileName):
    folder = path + "range_pathLength/"
    plt.switch_backend('Agg') 
    plt.figure(figsize=(8,6))                   # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(length,range,'b',label="ours") 
    plt.ylabel("explored range")
    plt.xlabel('path length')
    plt.legend()        #个性化图例（颜色、形状等）
    plt.savefig(folder+fileName+".jpg")

def TestPolicy(env, model, log=True, seed=None, startPoint=None):

    LogFolderInit()
    observation,_ = env.reset(seed=seed,startPoint=startPoint)
    Done = False
    state = None
    action ,state= model.predict(observation,state,deterministic=False)
    pathLength = 0
    rewardSum = 0
    cnt = 1
    rangeList = [0]
    lengthList = [0]
    while not Done:
        action = int(action)
        observation,reward,Done,_,_ = env.step(action)
        if log:
            print(action,reward,cnt)
        if action <= 3 :
            pathLength += 3
        else:
            pathLength += 3 * 1.414
        rewardSum += reward
        cnt += 1
        rangeList.append(env.exploredRange)
        lengthList.append(pathLength)
        if not Done:
            action ,state = model.predict(observation,state,deterministic=False)
    return rewardSum,pathLength,rangeList,lengthList,cnt

def multiTest(testTimes):
    env = GridWorldEnv(render=False)
    model = PPO.load('saved_model/hole_20_3')
    rewardList = []
    pathLengthList = []
    seedList = []
    startPoints = []
    time = 0
    while True:
        rewardSum,pathLength,rangeList,lengthList,cnt = TestPolicy(env,model,log=False)
        print(rewardSum,cnt,pathLength)
        if rewardSum > 7:
            seed = env.polygonFile.split(".")[0]
            rewardList.append(rewardSum)
            pathLengthList.append(pathLength)
            seedList.append(seed)
            startPoints.append(env.path[0])
            time+=1
        if time > testTimes:
            break

    print('pathLength: ',pathLengthList)
    print('seedList: ',seedList)
    print("pathAverage",sum(pathLengthList)/testTimes)
    print("startPoint",startPoints)

def seedTest():
    env = GridWorldEnv(render=True)
    model = PPO.load('saved_model/hole_20_3_1')
    seeds = ['862', '9798', '6769', '7274', '5439', '9675', '2342', '373', '2590', '8304', '5909', '6874', '7955', '9272', '2801', '4309', '5077', '772', '3111', '8369', '926']
    startPoints = [(182, 46), (215, 173), (43, 60), (134, 102), (176, 133), (204, 85), (168, 84), (128, 183), (154, 57), (61, 142), (137, 117), (83, 130), (83, 87), (234, 74), (243, 79), (66, 68), (201, 47), (154, 130), (248, 143), (80, 80), (179, 132)]
    rewardList = []
    pathLengthList = []
    for i in range(len(seeds)):
        rewardSum = 0
        seed = seeds[i]
        startPoint = startPoints[i]
        while rewardSum < 7:
            rewardSum,pathLength,rangeList,lengthList,cnt = TestPolicy(env,model,log=False,seed=seed,startPoint=startPoint)
            print(rewardSum,cnt,pathLength)
        DrawFullPath(env,seed)
        rewardList.append(rewardSum)
        pathLengthList.append(pathLength)

    print('pathLength: ',pathLengthList)
    print('seedList: ',seeds)
    print("pathAverage",sum(pathLengthList)/len(seeds))
    print("startPoint",startPoints)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == 'seed':
            seedTest()
        else:
            multiTest(int(sys.argv[1]))
    else:
        env = GridWorldEnv(render=True)
        model = PPO.load('saved_model/hole_20_3_1')
        rewardList = []
        pathLengthList = []
        exploredRangeList = []
        seedList = []

        while True:
            rewardSum,pathLength,rangeList,lengthList,cnt = TestPolicy(env,model)
            seed = env.polygonFile.split(".")[0]
            PlotRangeLength(rangeList,lengthList,seed)
            DrawFullPath(env,seed)
            MakeGif("tmp0","result0.gif",cnt)
            MakeGif("tmp1","result1.gif",cnt)
            MakeGif("tmp2","result2.gif",cnt)
            
            print("gif saved!")
            rewardList.append(rewardSum)
            pathLengthList.append(pathLength)
            seedList.append(seed)
            cmd = input()
            if cmd == 'q':
                break

        print('reward: ',rewardList)
        print('pathLength: ',pathLengthList)
        print('seedList: ',seedList)


