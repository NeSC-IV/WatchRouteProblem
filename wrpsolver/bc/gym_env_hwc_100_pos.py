# import gym
# from gym import spaces
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import cv2
import math
import shapely
from shapely.validation import make_valid

import os
import random
from random import choice
import json
import copy
from ..Test.draw_pictures import DrawMultiline,DrawPolygon,DrawPoints
from ..MACS.polygons_coverage import FindVisibleRegion,SelectMaxPolygon
from ..Test.vis_maps import GetPolygon
STEP = 3
PIC_SIZE = 100
PIC_DIR_NAMES = None
MAXSTEP = 400
DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/optimal_path_60_3/"
IMAGE = np.empty((PIC_SIZE, PIC_SIZE,1), dtype=np.uint8)
IMAGE.fill(150)
RANGE = 64
LOCAL_SHAPE = int((RANGE*2+1)/2)
LOCAL_SHAPE1 = int(32*2+1)
GLOBAL_SHAPE = 200
PATH_LEN = 30

class GridWorldEnv(gym.Env):

    def __init__(self, polygon=None, startPos=None, render = False):
        self.render_mode = None
        self.polygon = polygon
        self.pos = startPos
        # self.polygonInited = True if polygon is not None else False
        # self.startPosInited = True if startPos is not None else False
        self.observationPolygon = None
        self.observation = None
        self.globalObs = None
        self.localObs = None
        self.localObs1 = None
        self.localObs2 = None
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0
        self.render = render

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, PIC_SIZE, shape=(12,), dtype=np.float32),
                "localImage":spaces.Box(0, 255, shape=(4,STEP*2), dtype=np.float16),
                "localImage1":spaces.Box(0, 255, shape=(LOCAL_SHAPE1,LOCAL_SHAPE1,1), dtype=np.uint8),
                "localImage2":spaces.Box(0, 255, shape=(LOCAL_SHAPE,LOCAL_SHAPE,1), dtype=np.uint8),
                # "globalImage":spaces.Box(0, 255, shape=(GLOBAL_SHAPE,GLOBAL_SHAPE,1), dtype=np.uint8),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([STEP, 0]),
            1: np.array([-STEP, 0]),
            2: np.array([0, STEP]),
            3: np.array([0, -STEP]),
            # 4: np.array([STEP, STEP]),
            # 5: np.array([-STEP, -STEP]),
            # 6: np.array([-STEP, STEP]),
            # 7: np.array([STEP, -STEP]),
        }
    def _getObservation(self,pos):
        image = self.initImage.copy()
        point = shapely.Point((pos[0],pos[1]))
        polygon = self.polygon
        visiblePolygon = self.observationPolygon
        result = True
        stepVisiblePolygon = shapely.Point(0,0)

        # try:
        if self.stepCnt > 0 and self.MoveOutOfRange():
            visiblePolygon = self.observationPolygon
            result = False
        else:
            if(visiblePolygon == None):
                stepVisiblePolygon = FindVisibleRegion(polygon=polygon,watcher = point, d = 60, useCPP=True)
                visiblePolygon = stepVisiblePolygon
            else:
                stepVisiblePolygon = FindVisibleRegion(polygon=polygon,watcher = point, d = 60, useCPP=True)
                visiblePolygon = visiblePolygon.union(stepVisiblePolygon)
                visiblePolygon = SelectMaxPolygon(visiblePolygon.simplify(0.5,False))

        if(visiblePolygon == None):
            print("visiblePolygon get failed")
            return False
        bufferedVisiblePolygon = visiblePolygon.buffer(1)
        if not bufferedVisiblePolygon.is_valid:
            bufferedVisiblePolygon = make_valid(bufferedVisiblePolygon)
        obstacle = (bufferedVisiblePolygon.intersection(self.o)).simplify(0.1,False)
        bufferedObstacle = obstacle.buffer(2,join_style=2)
        if not bufferedObstacle.is_valid:
            bufferedObstacle = make_valid(bufferedObstacle)
        frontier = visiblePolygon.boundary.difference(bufferedObstacle)
        frontierList = GetFrontierList(frontier)
        agent = list(self.pos) + frontierList
        
        DrawMultiline(image,visiblePolygon, (255))
        DrawMultiline(image,stepVisiblePolygon, (220))
        DrawMultiline(image,obstacle,color = (0))
        for p in self.path[:]:
            x = p[0]
            y = p[1]
            image[y][x] = 80
        self.localObs = self.GetObscatleMap(image,pos[0],pos[1])
        self.localObsImage = self.GetLocalImage(image,pos[0],pos[1],STEP*2)
        DrawMultiline(image,frontier, color=(180))
        DrawPoints(image,point.x,point.y,(30),2)
        self.localObs1 = self.GetLocalImage(image,pos[0],pos[1],32)
        self.localObs2 = self.GetLocalImage(image,pos[0],pos[1],RANGE)
        self.localObs2 = cv2.resize(self.localObs2,(LOCAL_SHAPE,LOCAL_SHAPE),interpolation = cv2.INTER_NEAREST)
        self.observationPolygon = visiblePolygon
        self.image = image
        self.observation = {"agent":np.array(agent),"localImage":np.array(self.localObs/255.0),"localImage1":self.localObs1.reshape(LOCAL_SHAPE1,LOCAL_SHAPE1,1),
                            "localImage2":self.localObs2.reshape(LOCAL_SHAPE,LOCAL_SHAPE,1)}
        if self.render:
            savePath = os.path.dirname(os.path.abspath(__file__))+'/../../render_saved/'
            cv2.imwrite(savePath+'tmp/'+str(self.stepCnt)+'.png',self.image)
            cv2.imwrite(savePath+'tmp1/'+str(self.stepCnt)+'.png',self.localObsImage)
            cv2.imwrite(savePath+'tmp2/'+str(self.stepCnt)+'.png',self.localObs1)
            cv2.imwrite(savePath+'tmp3/'+str(self.stepCnt)+'.png',self.localObs2)
            image = self.initImage.copy()
            image.fill(0)
            DrawMultiline(image,self.polygon, (255))
            cv2.imwrite(savePath+'polygon.png',image)
        # except Exception as e:
        #     print(e)
        #     return False
        # else:
        #     return result
        return result
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, seed=None, polygon=None, startPoint = None):
        self.observationPolygon = None
        self.obstacle = None
        self.observation = self.observation_space.sample()
        self.globalObs = None
        self.localObs = None
        self.localObs1 = None
        self.localObs2 = None
        self.unknownGridNum = None
        self.exploredRange = None
        self.path = []
        self.stepCnt = 0
        self.polygon = polygon
        self.pos = startPoint
        self.initImage = None

        if not self.polygon:
            self.polygon = RandomGetPolygon()
        if not self.pos:
            self.pos = GetStartPoint(self.polygon)
        self.polygon = self.polygon.buffer(0.7,join_style=2)
        minx, miny, maxx, maxy = self.polygon.bounds
        self.maxx = math.ceil(maxx/10)*10
        self.maxy = math.ceil(maxy/10)*10
        self.initImage = np.zeros((self.maxy, self.maxx, 1), dtype=np.uint8)
        self.initImage.fill(150)
        self.image = self.initImage
        PIC_SIZE = (max(maxx,maxy))+10
        self.o = shapely.Polygon([(0,0),(PIC_SIZE,0),(PIC_SIZE,PIC_SIZE),(0,PIC_SIZE)]).difference(self.polygon)


        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        # self.unknownGridNum = CountUnkown(self.image)
        self.exploredRange = self.observationPolygon.area/self.polygon.area

        return self.observation,None
    def step(self,action):

        #定义变量
        timePunishment = 0
        reward = 0
        info = self._get_info()
        gamma = 1
        Done = True

        #更新位置
        direction = self._action_to_direction[action]
        self.action = action
        self.pos = (self.pos+direction).tolist()
        self.stepCnt += 1

        #agent步数是否到达上限
        if self.stepCnt >= MAXSTEP:
            self._getObservation(self.pos)

        #更新观测失败
        elif not self._getObservation(self.pos):
            reward = -1

        else:
            Done = False

        #计算奖励
        if not Done:

            tempExploredRange = self.observationPolygon.area/self.polygon.area
            exploreReward = (tempExploredRange - self.exploredRange)
            exploreReward = exploreReward if exploreReward > 1e-4 else 0
            self.exploredRange = tempExploredRange
            
            repeatPunishment = - 0.0 if (self.pos in self.path[:]) else 0
            
            if(self.observationPolygon.area/self.polygon.area > 0.95):
                Done = True
                reward+=1
            reward += float(exploreReward+timePunishment+repeatPunishment)

        #更新路径
        self.path.append(copy.copy(self.pos))
        return self.observation, reward*gamma, Done ,False, info
    
    def MoveOutOfRange(self):
        oneStep = self._action_to_direction[self.action]/3
        pos = np.array(self.pos)
        for i in range(STEP):
            pos = (pos - (oneStep * i)).astype(np.int32)
            x = pos[0]
            y = pos[1]
            p = shapely.Point(x,y)
            #out of boundary
            if (x >= self.maxx) or (y >= self.maxy) or (x <= 0) or (y <= 0):
                return True
            # obstacle
            if (self.image[y][x] == 0)  or (not self.polygon.contains(p)):
            # if (self.image[y][x] == 0) :
                return True
        return False

    def GetLocalImage(self,image,x,y,_range=1):
        _range = _range
        paddleSize = 128
        x = x + paddleSize
        y = y + paddleSize
        y_low = max((y-_range),0)
        y_high = min((y+_range),self.maxy+paddleSize*2)
        x_low = max((x-_range),0)
        x_high = min((x+_range),self.maxx+paddleSize*2)
        newImage = cv2.copyMakeBorder(image,paddleSize,paddleSize,paddleSize,paddleSize,cv2.BORDER_CONSTANT,value=0)
        newImage = newImage[y_low:y_high+1,[row for row in range(x_low,x_high+1)]]
        return newImage
    
    def GetObscatleMap(self, image, x , y):
        # image = np.zeros((image.shape[0],image.shape[1]))
        # image[y][x] = 150
        result = np.zeros((4,STEP*2))
        paddleSize = STEP * 4
        x = x + paddleSize
        y = y + paddleSize
        newImage = cv2.copyMakeBorder(image,paddleSize,paddleSize,paddleSize,paddleSize,cv2.BORDER_CONSTANT,value=0)
        # newImage[y,[row for row in range(x + 1, x + STEP*2 + 1)]] = 255
        # newImage[y,[row for row in range(x - STEP*2, x)]] = 255
        # newImage[(y + 1) : (y + STEP*2 + 1), x] = 255
        # newImage[(y - STEP*2) : y, x] = 255

        result[0] = newImage[y,[row for row in range(x + 1, x + STEP*2 + 1)]]
        result[1] = newImage[y,[row for row in range(x - STEP*2, x)]]
        result[2] = newImage[(y + 1) : (y + STEP*2 + 1), x]
        result[3] = newImage[(y - STEP*2) : y, x]
        return result
        


def RandomGetPolygon():
    global PIC_DIR_NAMES
    while True:        
        if not PIC_DIR_NAMES:
            PIC_DIR_NAMES = os.listdir(DIR_PATH)
        testJsonDir = DIR_PATH + choice(PIC_DIR_NAMES)
        with open(testJsonDir) as json_file:
            json_data = json.load(json_file)
        polygon = shapely.Polygon(json_data['polygon'])
        # image = IMAGE.copy()
        # DrawMultiline(image,polygon, color=(255))
        # cv2.imwrite('test.png',image)
        if polygon.is_valid and polygon.area < 30000 and polygon.area > 8000:
            return polygon
        
# def RandomGetPolygon():
#     while True:
#         seed = np.random.randint(0,30000)
#         pointList,_,roomNum = GetPolygon(seed)
#         polygon = shapely.Polygon(pointList).simplify(0.5,True).buffer(-0.7,join_style=2)
#         if(roomNum <= 3):
#             continue
#         elif(type(polygon) != shapely.Polygon):
#             continue
#         elif polygon.is_valid:
#             return polygon
        
def GetStartPoint(polygon):
    temppolygon = polygon.buffer(-2,join_style=2)
    minx, miny, maxx, maxy = temppolygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if temppolygon.contains(p):
            return (int(p.x),int(p.y))

def CountUnkown(image):
    _,thresh=cv2.threshold(image,149,255,cv2.THRESH_BINARY)
    cnt1 = cv2.countNonZero(thresh) #cnt1 = freespace + frontier + unknown
    _,thresh=cv2.threshold(image,151,255,cv2.THRESH_BINARY)
    cnt2 = cv2.countNonZero(thresh) #cnt1 = freespace + frontier
    return cnt1 - cnt2

def GetFrontierList(frontier):
    ft_list = []
    ft_num = 5 * 2
    def GetFrontierListBaseType(frontier, ft_list):
        try:
            if(type(frontier) == shapely.Point):
                ft_list.extend([frontier.x,frontier.y])
            elif(type(frontier) == shapely.LineString or type(frontier) == shapely.Polygon or type(frontier) == shapely.LinearRing):
                point = frontier.centroid
                ft_list.extend([point.x,point.y])
        except Exception as e:
            pass
    
    if (type(frontier) == shapely.Point or type(frontier) == shapely.LinearRing or 
        type(frontier) == shapely.LineString or type(frontier) == shapely.Polygon):
        GetFrontierListBaseType(frontier,ft_list)
    else:
        for geometry in list(frontier.geoms):
            GetFrontierListBaseType(geometry,ft_list)

    if (len(ft_list) == 0):
        ft_list.extend(0 for _ in range(ft_num))
    elif (len(ft_list) < ft_num):
        while(len(ft_list) < ft_num):
            ft_list.append(ft_list[-2])
            ft_list.append(ft_list[-2])
    else:
        ft_list = ft_list[:ft_num]

    return ft_list
        