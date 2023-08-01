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
from ..Test.draw_pictures import DrawMultiline,DrawPolygon,DrawPoints
from ..MACS.polygons_coverage import FindVisibleRegion,SelectMaxPolygon

STEP = 1
PIC_SIZE = 100
PIC_DIR_NAMES = None
MAXSTEP = 400
DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/pic_data_picsize100_new/"
IMAGE = np.empty((PIC_SIZE, PIC_SIZE,1), dtype=np.uint8)
IMAGE.fill(150)

class GridWorldEnv(gym.Env):

    def __init__(self, polygon=None, startPos=None):
        self.render_mode = None
        self.polygon = polygon
        self.pos = startPos
        self.polygonInited = True if polygon is not None else False
        self.startPosInited = True if startPos is not None else False
        self.observationPolygon = None
        self.observation = None
        self.globalObs = None
        self.localObs = None
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0

        self.observation_space = spaces.Box(low=0, high=255, shape=(PIC_SIZE, PIC_SIZE, 3), dtype=np.uint8)

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
        image_ft = IMAGE.copy()
        image_ft.fill(255)
        point = shapely.Point((pos[0],pos[1]))
        polygon = self.polygon
        visiblePolygon = self.observationPolygon

        try:
            if not self.polygon.covers(point):
                print("polygon not contains point")
                return False
            
            if(visiblePolygon == None):
                visiblePolygon = FindVisibleRegion(polygon=polygon,watcher = point,d= 800,useCPP=True)
            else:
                visiblePolygon = SelectMaxPolygon(visiblePolygon.union(FindVisibleRegion(polygon=polygon,watcher = point,d= 800,useCPP=True)))
            if(visiblePolygon == None):
                print("visiblePolygon get failed")
                return False
            visiblePolygon = visiblePolygon.simplify(0.05,False)
            obcastle = visiblePolygon.buffer(2).intersection(self.o)
            frontier = visiblePolygon.boundary.difference(obcastle.buffer(4))

            DrawMultiline(image,obcastle,color = (0))
            DrawMultiline( image,visiblePolygon, (255))
            DrawMultiline(image,frontier, color=(200))
            for p in self.path:
                x = p[0]
                y = p[1]
                image[y][x] = 80
            DrawPoints(image,point.x,point.y,(30),-1)

            DrawMultiline(image_ft,frontier, color=(150))
            DrawPoints(image_ft,point.x,point.y,(30),-1)
            self.observationPolygon = visiblePolygon
            self.globalObs = image
            self.localObs = GetLocalImage(image,pos[0],pos[1])
            self.observation = cv2.merge([self.globalObs, self.localObs,image_ft])
        except Exception as e:
            print(e)
            return False
        else:
            return True
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, seed=None):
        self.observationPolygon = None
        self.observation = np.zeros((PIC_SIZE, PIC_SIZE,3), dtype=np.uint8)
        self.globalObs = None
        self.localObs = None
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0

        if not self.polygonInited:
            self.polygon = RandomGetPolygon().simplify(0.05, preserve_topology=False)
        if not self.startPosInited:
            self.pos = GetStartPoint(self.polygon)
        self.o = shapely.Polygon([(0,0),(PIC_SIZE,0),(PIC_SIZE,PIC_SIZE),(0,PIC_SIZE)]).difference(self.polygon).buffer(-0.5)


        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        self.unknownGridNum = CountUnkown(self.globalObs)

        return self.observation,None
    
    def step(self,action):

        #定义变量
        timePunishment = -0.001
        reward = 0
        info = self._get_info()
        gamma = 1
        Done = True

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
            reward -= 1
            # Done = False


        #更新观测失败
        elif not self._getObservation(self.pos):
            reward = -1

        else:
            Done = False

        #计算奖励
        if not Done:
            tempGridCnt = CountUnkown(self.globalObs)
            exploreReward = max((self.unknownGridNum - tempGridCnt) * 0.0002,0)
            exploreReward = min(exploreReward,0.5)
            self.unknownGridNum = tempGridCnt
            
            if(self.observationPolygon.area/self.polygon.area > 0.95):
                Done = True
                reward+=1

            reward += float(exploreReward+timePunishment)

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
        if (self.globalObs[y][x] == 0) or (not self.polygon.contains(p)):
            return True
        return False

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
    # print(newImage.shape)
    # newImage = cv2.resize(newImage,(paddleSize,paddleSize),interpolation = cv2.INTER_NEAREST)
    return newImage

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