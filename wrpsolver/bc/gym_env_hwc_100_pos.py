# import gym
# from gym import spaces
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import cv2
import math
import shapely
import os
import random
from random import choice
import json
import copy
from ..Test.draw_pictures import DrawMultiline,DrawPolygon,DrawPoints
from ..MACS.polygons_coverage import FindVisibleRegion,SelectMaxPolygon

STEP = 1
PIC_SIZE = 100
PIC_DIR_NAMES = None
MAXSTEP = 1000
DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/optimal_path/"
IMAGE = np.empty((PIC_SIZE, PIC_SIZE,1), dtype=np.uint8)
IMAGE.fill(150)
RANGE = 48
LOCAL_SHAPE = (RANGE*2+1)*1
PATH_LEN = 20

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
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0
        self.render = render

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, PIC_SIZE, shape=(12,), dtype=np.float32),
                "localImage":spaces.Box(0, 255, shape=(5,5), dtype=np.uint8),
                "globalImage":spaces.Box(0, 255, shape=(LOCAL_SHAPE,LOCAL_SHAPE,1), dtype=np.uint8),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([STEP, 0]),
            1: np.array([-STEP, 0]),
            2: np.array([0, STEP]),
            3: np.array([0, -STEP]),
        }
    def _getObservation(self,pos):
        image = IMAGE.copy()
        point = shapely.Point((pos[0],pos[1]))
        polygon = self.polygon
        visiblePolygon = self.observationPolygon
        result = True
        stepVisiblePolygon = shapely.Point(0,0)

        try:
            if self.MoveOutOfRange() or (not self.polygon.contains(point)):
                visiblePolygon = self.observationPolygon
                result = False
            else:
                if(visiblePolygon == None):
                    stepVisiblePolygon = FindVisibleRegion(polygon=polygon,watcher = point, d = 32,useCPP=True)
                    visiblePolygon = stepVisiblePolygon
                else:
                    stepVisiblePolygon = FindVisibleRegion(polygon=polygon,watcher = point, d = 32,useCPP=True)
                    visiblePolygon = SelectMaxPolygon(visiblePolygon.union(stepVisiblePolygon))
            if(visiblePolygon == None):
                print("visiblePolygon get failed")
                return False
            visiblePolygon = visiblePolygon.simplify(0.05,False)
            obstacle = (visiblePolygon.buffer(2,join_style=2).intersection(self.o))
            frontier = visiblePolygon.boundary.difference(obstacle.buffer(2,join_style=2))
            frontierList = GetFrontierList(frontier)
            agent = list(self.pos) + frontierList
            
            DrawMultiline(image,obstacle,color = (0))
            DrawMultiline(image,visiblePolygon, (255))
            DrawMultiline(image,stepVisiblePolygon, (230))
            for p in self.path[-PATH_LEN:]:
                x = p[0]
                y = p[1]
                image[y][x] = 80
            self.localObs = GetLocalImage(image,pos[0],pos[1],2)
            DrawMultiline(image,frontier, color=(200))
            DrawPoints(image,point.x,point.y,(30),-1)
            self.globalObs = GetLocalImage(image,pos[0],pos[1],RANGE)
            self.globalObs = cv2.resize(self.globalObs,(LOCAL_SHAPE,LOCAL_SHAPE),interpolation = cv2.INTER_NEAREST)
            self.observationPolygon = visiblePolygon
            for p in self.path[:-PATH_LEN]:
                x = p[0]
                y = p[1]
                image[y][x] = 80
            self.image = image
            self.observation = {"agent":np.array(agent),"localImage":np.array(self.localObs),"globalImage":self.globalObs.reshape(LOCAL_SHAPE,LOCAL_SHAPE,1)}
            # self.observation = {"agent":np.array(agent),"localImage":np.array(self.localObs),"globalImage":image.reshape(100,100,1)}
            if self.render:
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/'+str(self.stepCnt)+'.png',self.image)
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/'+str(self.stepCnt)+'.png',self.globalObs)
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/'+str(self.stepCnt)+'.png',self.localObs)
        except Exception as e:
            print(e)
            return False
        else:
            return result
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, seed=None, polygon=None, startPoint = None):
        self.observationPolygon = None
        self.observation = self.observation_space.sample()
        self.image = IMAGE.copy()
        self.globalObs = None
        self.localObs = None
        self.unknownGridNum = None
        self.exploredRange = None
        self.path = []
        self.stepCnt = 0
        self.polygon = polygon
        self.pos = startPoint

        if not self.polygon:
            self.polygon = RandomGetPolygon()
        if not self.pos:
            self.pos = GetStartPoint(self.polygon)
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
            exploreReward = (tempExploredRange - self.exploredRange)*2
            self.exploredRange = tempExploredRange
            
            repeatPunishment = - 0.001 if (self.pos in self.path[-PATH_LEN:]) else 0
            
            if(self.observationPolygon.area/self.polygon.area > 0.9):
                Done = True
                reward+=1
            reward += float(exploreReward+timePunishment+repeatPunishment)

        #更新路径
        self.path.append(copy.copy(self.pos))
        return self.observation, reward*gamma, Done ,False, info
    
    def MoveOutOfRange(self):
        x = self.pos[0]
        y = self.pos[1]
        p = shapely.Point(x,y)
        #out of boundary
        if (x >= PIC_SIZE) or (y >= PIC_SIZE) or (x <= 0) or (y <= 0):
            return True
        # obstacle
        if (self.image[y][x] == 0) or (not self.polygon.contains(p)):
            return True
        return False

def GetLocalImage(image,x,y,_range=1):
    _range = _range
    paddleSize = 80
    x = x + paddleSize
    y = y + paddleSize
    y_low = max((y-_range),0)
    y_high = min((y+_range),PIC_SIZE+paddleSize*2)
    x_low = max((x-_range),0)
    x_high = min((x+_range),PIC_SIZE+paddleSize*2)
    newImage = cv2.copyMakeBorder(image,paddleSize,paddleSize,paddleSize,paddleSize,cv2.BORDER_CONSTANT,value=0)
    newImage = newImage[y_low:y_high+1,[row for row in range(x_low,x_high+1)]]
    return newImage

def RandomGetPolygon():
    global PIC_DIR_NAMES
    while True:        
        if not PIC_DIR_NAMES:
            PIC_DIR_NAMES = os.listdir(DIR_PATH)
        testJsonDir = DIR_PATH + choice(PIC_DIR_NAMES)
        with open(testJsonDir) as json_file:
            json_data = json.load(json_file)
        polygon = shapely.Polygon(json_data['polygon'])
        image = IMAGE.copy()
        DrawMultiline(image,polygon, color=(255))
        cv2.imwrite('test.png',image)
        if polygon.is_valid:
            return polygon
        
def GetStartPoint(polygon):
    temppolygon = polygon.buffer(-STEP*2,join_style=2)
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
        