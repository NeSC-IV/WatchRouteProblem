import os
import cv2
import sys
import shutil
import matplotlib.pyplot as plt
import math
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
            action ,state = model.predict(observation,state,deterministic=True)
    return rewardSum,pathLength,rangeList,lengthList,cnt

def multiTest(testTimes):
    env = GridWorldEnv(render=False)
    model = PPO.load('saved_model/hole_20_3_2')
    rewardList = []
    pathLengthList = []
    seedList = []
    startPoints = []
    time = 0
    while True:
        rewardSum,pathLength,rangeList,lengthList,cnt = TestPolicy(env,model,log=False)
        seed = env.polygonFile.split(".")[0]
        if rewardSum > 7:
            rewardList.append(rewardSum)
            pathLengthList.append(pathLength)
            seedList.append(seed)
            startPoints.append(env.path[0])
            time+=1
        print(seed,rewardSum,cnt,pathLength)
        if time > testTimes:
            break

    print('pathLength: ',pathLengthList)
    print('seedList: ',seedList)
    print("pathAverage",sum(pathLengthList)/testTimes)
    print("startPoint",startPoints)

def seedTest():
    env = GridWorldEnv(render=True)
    model = PPO.load('saved_model/hole_20_3_2')
    seeds = ['3072', '8889', '5676', '9410', '5987', '7430', '2922', '5623', '4253', '7655', '3746', '4198', '6508', '2100', '7942', '6508', '4239', '1123', '3240', '9589', '7715']
    seeds = ['3746']

    startPoints = [[109, 166], [252, 104], [81, 76], [202, 122], [108, 61], [24, 148], [159, 90], [139, 189], [251, 141], [50, 92], [218, 125], [158, 78], [86, 69], [167, 70], [213, 113], [123, 148], [200, 93], [212, 57], [171, 68], [45, 88], [179, 43]]
    startPoints = [[218, 125]]
    # seeds = ['1861', '7665', '5271', '4397', '4082', '5326', '4667', '3904', '9561', '8296', '9944', '8092', '3955', '72', '327', '8884', '3604', '5986', '930', '2340', '2713']
    # startPoints = [[190, 186], [124, 114], [67, 37], [172, 147], [72, 125], [31, 131], [94, 57], [211, 183], [176, 184], [253, 81], [44, 158], [174, 51], [102, 50], [77, 121], [77, 84], [177, 119], [56, 110], [73, 204], [153, 61], [116, 186], [86, 185]]
    rewardList = []
    pathLengthList = []
    for i in range(len(seeds)):
        length = 0
        rewardSum = 0
        seed = seeds[i]
        startPoint = startPoints[i]
        while rewardSum < 7:
            rewardSum,pathLength,rangeList,lengthList,cnt = TestPolicy(env,model,log=False,seed=seed,startPoint=startPoint)
            print(seed,rewardSum,cnt,pathLength)
        # print(env.path)
        pathList = []
        for pos in env.path:
            if pos not in pathList:
                pathList.append(pos)
        for i in range(1,len(pathList)):
            dx = pathList[i][0] - pathList[i-1][0]
            dy = pathList[i][1] - pathList[i-1][1]
            length += math.sqrt(dx*dx+dy*dy)
        print(seed,rewardSum,cnt,length)
        print(env.path)
        DrawFullPath(env,seed)
        rewardList.append(rewardSum)
        pathLengthList.append(length)

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


