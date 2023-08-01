import gym
from gym import spaces
import numpy as np
import cv2
import shapely
import os
import random
from random import choice
import json
import copy
from ..Test.draw_pictures import DrawMultiline,DrawPolygon,DrawPoints
from ..MACS.polygons_coverage import FindVisibleRegion
step = 1
pic_size = 100
picDirNames = None
dirPath = os.path.dirname(os.path.abspath(__file__))+"/../Test/pic_data_picsize100_new/"

def getStartPoint(polygon):
    temppolygon = polygon.buffer(-step)
    minx, miny, maxx, maxy = temppolygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if temppolygon.covers(p):
            return (int(p.x),int(p.y))


def countUnkown(image):
    ret,thresh=cv2.threshold(image,254,255,cv2.THRESH_BINARY_INV)
    cnt = cv2.countNonZero(thresh)
    return cnt
def Polygon2Gird(polygon):

    grid = np.zeros((pic_size, pic_size,1), dtype=np.uint8)
    points = list(polygon.exterior.coords)
    points = np.array(points)
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        grid = cv2.fillPoly(grid, [points], 255)
    else:
        grid = cv2.fillPoly(grid, points, 255)

    return grid
class GridWorldEnv(gym.Env):

    def __init__(self, polygon=None, startPos=None):
        self.polygon = polygon
        self.startPos = startPos
        self.pos = startPos
        self.polygonInited = True if polygon is not None else False
        self.posInited = True if startPos is not None else False
        self.observationPolygon = shapely.Point(1,1)
        self.observation = None
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0

        self.observation_space = spaces.Box(low=0, high=255, shape=(pic_size, pic_size, 1), dtype=np.uint8)

        self.action_space = spaces.Discrete(8)

        self._action_to_direction = {
            0: np.array([step, 0]),
            1: np.array([-step, 0]),
            2: np.array([0, step]),
            3: np.array([0, -step]),
            4: np.array([step, step]),
            5: np.array([-step, -step]),
            6: np.array([-step, step]),
            7: np.array([step, -step]),
        }

    def _getObservation(self,pos):
        image = np.empty((pic_size, pic_size,1), dtype=np.uint8)
        image.fill(150)
        point = shapely.Point((pos[0],pos[1]))
        visiblePolygon = self.observationPolygon

        try:
            if not self.polygon.covers(point):
                print("polygon not contains point")
                return False
            
            if(visiblePolygon == None):
                visiblePolygon = FindVisibleRegion(polygon=self.polygon,watcher = point,d= 80,useCPP=True)
            else:
                visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon=self.polygon,watcher = point,d= 80,useCPP=True))
            visiblePolygon = visiblePolygon.simplify(0.1, preserve_topology=False)
            if(visiblePolygon == None):
                return False
            
            obcastle = self.o.intersection(visiblePolygon.buffer(0.5))
            DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)
            for p in self.path:
                x = p[0]
                y = p[1]
                image[y][x] = 80
            DrawPoints(image,point.x,point.y,(30),-1)
            DrawMultiline(image,obcastle,color = (0))

            self.observationPolygon = visiblePolygon
            # self.observation = self.image.copy().reshape(pic_size,pic_size,1)
            self.observation = image
        except Exception as e:
            print(e)
            self.image = np.zeros((pic_size, pic_size,1), dtype=np.uint8)
            return False
        else:
            return True
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, polygon=None, startPos=None, seed=None):
        global picDirNames
        self.observationPolygon = None
        self.observation = None
        self.path = []
        self.stepCnt = 0

        if (polygon):
            self.polygon = polygon
        elif not self.polygonInited:    
            while True:        
                if not picDirNames:
                    picDirNames = os.listdir(dirPath)
                    picDirNames.sort()
                testJsonDir = dirPath + choice(picDirNames) + '/data.json'
                with open(testJsonDir) as json_file:
                    json_data = json.load(json_file)
                self.polygon = shapely.Polygon(json_data['polygon'])
                if self.polygon.is_valid:
                    break
        self.polygon = self.polygon.simplify(0.05, preserve_topology=False)
        self.o = shapely.Polygon([(0,0),(pic_size,0),(pic_size,pic_size),((0,pic_size))]).difference(self.polygon)

        if (startPos):
            self.pos = startPos
        elif not self.posInited:
            self.pos = getStartPoint(self.polygon)
        else:
            self.pos = self.startPos


        try:
            self._getObservation(self.pos)
        except:
            print("getObservation failed")
        finally:
            self.path.append(copy.copy(self.pos))
            self.unknownGridNum = countUnkown(self.observation)

            return self.observation
    
    def step(self,action):

        #定义变量
        timePunishment = -0.0001
        maxStep = 400
        boundary = pic_size
        reward = 0
        info = self._get_info()
        gamma = 1

        #更新位置
        direction = self._action_to_direction[action]
        self.pos += direction
        self.stepCnt += 1

        #agent步数是否到达上限
        if self.stepCnt >= maxStep:
            reward = 0
            Done = True

        #agent是否移动到地图外
        elif abs(self.pos[1])>=boundary or abs(self.pos[0])>=boundary or (self.observation[self.pos[1]][self.pos[0]] == 0) or (not self.polygon.contains(shapely.Point((self.pos[0],self.pos[1])))):
            self.pos -= direction
            reward = -1
            Done = False

        #agent是否撞上障碍物
        else :
            result = self._getObservation(self.pos)
            if not result:
                reward = -1
                Done = True
            else:
                Done = False

        #计算奖励
        if not Done:
            tempGridCnt = countUnkown(self.observation)
            exploreReward = max((self.unknownGridNum - tempGridCnt) * 0.0005,0)
            exploreReward = min(exploreReward,0.5)
            self.unknownGridNum = tempGridCnt
            
            if(self.observationPolygon.area/self.polygon.area > 0.9):
                Done = True
            else:
                Done = False
            reward = reward + float(exploreReward+timePunishment)

        #更新路径
        self.path.append(copy.copy(self.pos))

        return self.observation, reward*gamma, Done ,info
