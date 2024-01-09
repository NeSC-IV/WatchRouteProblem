import os
import cv2
import math
import shutil
from PIL import Image
from wrpsolver.GTSP.astar.a_star import findPath
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv,DrawPolygon
def FrontierPolicy(frontier, agent, image, step, polygon=None):
    actionList = []
    actionDict = {(step,0):0,(-step,0):1,(0,step):2,(0,-step):3}
    furthestDistance = float('inf')
    furthestPath = None
    posX = agent[0]
    posY = agent[1]
    for i in range(0, len(frontier), 2):
        goalX = int(math.ceil(frontier[i])*step + posX)
        goalY = int(math.ceil(frontier[i+1])*step + posY)
        path, distance = findPath((posX , posY), (goalX , goalY), image, polygon)
        # if path == None or (len(path) == 0):
        if path != None and distance < furthestDistance and len(path) > 1:
            furthestDistance = distance
            furthestPath = path
    if furthestPath == None:
        print(agent,frontier)
        return None
    for i in range(1,len(furthestPath)):
        action = (furthestPath[i][0] -  posX, furthestPath[i][1] -  posY)
        action = actionDict[action]
        actionList.append(action)
        posX = furthestPath[i][0]
        posY = furthestPath[i][1]
    return actionList

        
if __name__ == '__main__':
    env = GridWorldEnv(render=True)
    observation,_ = env.reset()
    image = env.initImage.copy()
    image.fill(0)
    DrawPolygon(env.polygon.buffer(-2, join_style=2), 255, image, zoomRate = 1)
    shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/")
    shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/")
    shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/")
    os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/")
    os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/")
    os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/")
    Done = False
    state = None
    actionList = FrontierPolicy(env.frontierList, env.pos, image, 3, env.polygon)
    rewardSum = 0
    cnt = 1
    while not Done:
        for action in actionList:
            action = int(action)
            observation,reward,Done,_,_ = env.step(action)
            print(action,reward,cnt)
            rewardSum += reward
            cnt += 1
            if Done:
                break
        if not Done:
            actionList = FrontierPolicy(env.frontierList, env.pos, image, 3, env.polygon)
            if(actionList == None):
                break
    frames = []
    for pic in range(1,cnt):
        im = Image.open("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/" + str(pic)+".png")
        frames.append(im)
    frame_one = frames[0]
    frame_one.save("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result.gif", format="GIF", append_images=frames,
            save_all=True, duration=120, loop=0)
    
    frames = []
    for pic in range(1,cnt):
        im = Image.open("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp3/" + str(pic)+".png")
        frames.append(im)
    frame_one = frames[0]
    frame_one.save("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result1.gif", format="GIF", append_images=frames,
            save_all=True, duration=120, loop=0)
    
    frames = []
    for pic in range(1,cnt):
        im = Image.open("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/" + str(pic)+".png")
        frames.append(im)
    frame_one = frames[0]
    frame_one.save("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result2.gif", format="GIF", append_images=frames,
            save_all=True, duration=120, loop=0)

    print("gif saved!")