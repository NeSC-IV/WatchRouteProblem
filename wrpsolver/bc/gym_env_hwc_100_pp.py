# import gym
# from gym import spaces
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import cv2
import shapely
import os
import random
from random import choice
import json
import copy
import math
from ..Test.draw_pictures import DrawMultiline,DrawPolygon,DrawPoints,DrawSinglePoint

STEP = 1
PIC_SIZE = 100
PIC_DIR_NAMES = None
MAXSTEP = 400
DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/pic_data_picsize100_new/"
IMAGE = np.zeros((PIC_SIZE, PIC_SIZE,1), dtype=np.uint8)
class GridWorldEnv(gym.Env):

    def __init__(self, polygon=None, startPos=None, seed = None, channel = False, render = False):
        self.channel = channel
        self.render = render
        self.render_mode = None
        self.polygon = polygon
        self.pos = startPos
        self.polygonInited = True if polygon is not None else False
        self.startPosInited = True if startPos is not None else False
        self.observation = None
        self.path = []
        self.stepCnt = 0
        self.goal = None

        if self.channel:
            self.observation_space = spaces.Box(low=0, high=255, shape=(PIC_SIZE, PIC_SIZE, 2), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(PIC_SIZE, PIC_SIZE, 1), dtype=np.uint8)

        self.action_space = spaces.Discrete(8)

        self._action_to_direction = {
            0: np.array([STEP, 0]),
            1: np.array([-STEP, 0]),
            2: np.array([0, STEP]),
            3: np.array([0, -STEP]),
            4: np.array([STEP, STEP]),
            5: np.array([-STEP, -STEP]),
            6: np.array([-STEP, STEP]),
            7: np.array([STEP, -STEP]),
        }

    def _getObservation(self,pos):
        image = IMAGE.copy()
        point = shapely.Point((pos[0],pos[1]))

        try:
            # if not self.polygon.covers(point):
            #     print("polygon not contains point")
            #     return False
            
            DrawMultiline(image,self.polygon, color=(255))
            for p in self.path:
                x = p[0]
                y = p[1]
                image[y][x] = 80
            DrawSinglePoint(image,point.x,point.y,(30),2,1)
            DrawSinglePoint(image,self.goal[0],self.goal[1],(150),2,1)
            self.image = self.observation = image
            if self.channel:
                localImage = GetLocalImage(image,pos[0],pos[1])
                self.observation =  cv2.merge([image,localImage])
            if self.render:
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp/'+str(self.stepCnt)+'.png',image)
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp1/'+str(self.stepCnt)+'.png',localImage)
        except Exception as e:
            print(e)
            return False
        else:
            return True
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, seed=None):
        self.observation = self.observation_space.sample()
        self.image = IMAGE
        self.path = []
        self.stepCnt = 0

        if not self.polygonInited:
            self.polygon = RandomGetPolygon().simplify(0.05, preserve_topology=False)
        if not self.startPosInited:
            self.pos = GetStartPoint(self.polygon)

        self.goal = GetStartPoint(self.polygon)


        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        self.distance = math.hypot(self.pos[0]-self.goal[0], self.pos[1]-self.goal[1])

        return self.observation,None
    
    def step(self,action):

        #定义变量
        timePunishment = -0.000001
        reward = 0
        info = self._get_info()
        gamma = 1
        Done = True
        distance = math.hypot((self.pos[0]-self.goal[0]), (self.pos[1]-self.goal[1]))
        #更新位置
        direction = self._action_to_direction[action]
        self.pos += direction
        self.stepCnt += 1

        #agent步数是否到达上限
        if self.stepCnt >= MAXSTEP:
            pass
        #agent是否移动到地图外
        elif self.MoveOutOfRange():
            self.pos -= direction
            reward -= 2


        #更新观测失败
        elif not self._getObservation(self.pos):
            reward = -2

        else:
            Done = False

        #计算奖励
        if not Done:
            distanceReward = (self.distance - distance) * 0.02
            if distance < 3:
                reward += 5
                Done = True

            reward += float(distanceReward+timePunishment)
            self.distance = distance
        #更新路径
        self.path.append(copy.copy(self.pos))
        return self.observation, reward*gamma, Done ,False,info
    
    def MoveOutOfRange(self):
        x = self.pos[0]
        y = self.pos[1]
        p = shapely.Point(x,y)
        #out of boundary
        if (x >= PIC_SIZE) or (y >= PIC_SIZE) or (x <= 0) or (y <= 0):
            return True
        # obscatle
        if (self.image[y][x] == 0) or (not self.polygon.contains(p)):
            return True
        return False

def RandomGetPolygon():
    global PIC_DIR_NAMES
    while True:        
        if not PIC_DIR_NAMES:
            PIC_DIR_NAMES = os.listdir(DIR_PATH)
        testJsonDir = DIR_PATH + choice(PIC_DIR_NAMES) + '/data.json'
        with open(testJsonDir) as json_file:
            json_data = json.load(json_file)
        polygon = shapely.Polygon(json_data['polygon'])
        if polygon.is_valid:
            return polygon
        
def GetStartPoint(polygon):
    temppolygon = polygon.buffer(-STEP)
    minx, miny, maxx, maxy = temppolygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if temppolygon.covers(p):
            return (int(p.x),int(p.y))


def CountUnkown(image):
    ret,thresh=cv2.threshold(image,254,255,cv2.THRESH_BINARY_INV)
    cnt = cv2.countNonZero(thresh)
    return cnt

def GetLocalImage(image,x,y,_range=5):
    _range = _range
    paddleSize = 40
    x = x + paddleSize
    y = y + paddleSize
    y_low = max((y-_range),0)
    y_high = min((y+_range),PIC_SIZE+paddleSize*2)
    x_low = max((x-_range),0)
    x_high = min((x+_range),PIC_SIZE+paddleSize*2)

    newImage = cv2.copyMakeBorder(image,paddleSize,paddleSize,paddleSize,paddleSize,cv2.BORDER_CONSTANT,value=0)
    ret,newImage=cv2.threshold(newImage,31,255,cv2.THRESH_TRUNC)
    DrawPoints(newImage,x,y,(255),-1)
    newImage = newImage[y_low:y_high,[row for row in range(x_low,x_high)]]
    newImage = cv2.resize(newImage,(PIC_SIZE,PIC_SIZE),interpolation = cv2.INTER_NEAREST)
    return newImage