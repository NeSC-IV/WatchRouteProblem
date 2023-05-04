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
import time
from func_timeout import func_set_timeout, FunctionTimedOut
from ..Test.draw_pictures import DrawMultiline,DrawSinglePoint,DrawPolygon,DrawPoints
from ..MACS.polygons_coverage import FindVisibleRegion
from shapely.validation import make_valid
step = 1
grid_size = 200
picDirNames = None
dirPath = os.path.dirname(os.path.abspath(__file__))+"/../Test/pic_data/pic_data/"

def getStartPoint(polygon):
    temppolygon = polygon.buffer(-3)
    minx, miny, maxx, maxy = temppolygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if temppolygon.contains(p):
            return (int(p.x),int(p.y))

def ObscatlePunishMent(polygon,pos):
    point = shapely.Point(pos)
    #one step obscatle
    try:
        if not polygon.covers(point.buffer(1)):
            reward =  -0.05
            Done = False
        if not polygon.covers(point.buffer(1)):
            reward =  -0.05
            Done = False
        elif not polygon.covers(point.buffer(2)):
            reward =  -0.02
            Done = False
        elif not polygon.covers(point.buffer(3)):
            reward =  -0.01
            Done = False
        else:
            reward = 0
    except Exception as e:
        print(e)
        return 0
    else:
        return reward,Done

def countUnkown(image):
    ret,thresh=cv2.threshold(image,254,255,cv2.THRESH_BINARY_INV)
    cnt = cv2.countNonZero(thresh)
    return cnt

def Polygon2Gird(polygon):

    grid = np.zeros((grid_size, grid_size,1), dtype=np.uint8)
    points = list(polygon.exterior.coords)
    points = np.array(points)
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        grid = cv2.fillPoly(grid, [points], 255)
    else:
        grid = cv2.fillPoly(grid, points, 255)

    return grid

def Image2Observation(image,pos):
    observation = np.zeros((3,grid_size, grid_size), dtype=np.uint8)

    ret,thresh=cv2.threshold(image,1,255,1) #obscatle
    observation[0] = thresh.reshape(1,200,200)

    ret,thresh=cv2.threshold(image,151,255,4) #unknown region
    observation[1] = thresh.reshape(1,200,200)

    observation[2][pos[1]][pos[0]] = 30


    return observation

class GridWorldEnv(gym.Env):

    def __init__(self, polygon=None, startPos=None):
        self.polygon = polygon
        self.startPos = startPos
        self.pos = startPos
        self.polygonInited = True if polygon is not None else False
        self.posInited = True if startPos is not None else False
        self.observationPolygon = None
        self.observation = None
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,grid_size, grid_size), dtype=np.uint8)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
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

    @func_set_timeout(3)
    def _getObservation(self,pos):
        # 更新observationPolygon和observation
        image = np.empty((grid_size, grid_size,1), dtype=np.uint8)
        image.fill(150)
        self.observation = np.zeros((3,grid_size, grid_size), dtype=np.uint8)
        point = shapely.Point(pos)
        polygon = self.polygon
        visiblePolygon = self.observationPolygon

        try:
            if not self.polygon.contains(shapely.Point(pos)):
                return False
            if(visiblePolygon == None):
                visiblePolygon = FindVisibleRegion(polygon,point,800, True)
            else:
                visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon=polygon,watcher = point,d= 800,useCPP=True))
            if(visiblePolygon == None):
                return False
            obcastle = visiblePolygon.boundary.intersection(polygon.boundary.buffer(1))
            # unknownRegion = visiblePolygon.boundary.difference(polygon.boundary.buffer(1))
            # obcastle = visiblePolygon.boundary.difference(unknownRegion.buffer(1))

            DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)
            DrawMultiline(image,obcastle,color = (0))
            DrawSinglePoint(image,point.x,point.y,(30))
            for point in self.path:
                x = point[0]
                y = point[1]
                image[y][x] = 80

            self.observationPolygon = visiblePolygon
            self.image = image
            self.observation = Image2Observation(image,pos)
        except Exception as e:
            print(e)
            self.image = np.zeros((grid_size, grid_size,1), dtype=np.uint8)
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
                    picDirNames = picDirNames[:100]
                testJsonDir = dirPath + choice(picDirNames) + '/data.json'
                with open(testJsonDir) as json_file:
                    json_data = json.load(json_file)
                self.polygon = shapely.Polygon(json_data['polygon'])
                if self.polygon.is_valid:
                    break
        self.polygon = self.polygon.simplify(0.01, preserve_topology=True)

        if (startPos):
            self.pos = startPos
        elif not self.posInited:
            self.pos = getStartPoint(self.polygon)
        else:
            self.pos = self.startPos
        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        self.unknownGridNum = countUnkown(self.image)

        return self.observation
    def step(self,action):
        direction = self._action_to_direction[action]
        self.pos += direction
        info = self._get_info()
        self.stepCnt += 1
        reward = 0
        # obscatlePunishMent = ObscatlePunishMent(self.polygon,self.pos)
        obscatlePunishMent = 0
        if abs(self.pos[1])>=200 or abs(self.pos[0])>=200 or (self.image[self.pos[1]][self.pos[0]] == 0) or (not self.polygon.contains(shapely.Point(self.pos))):
            self.pos -= direction
            reward = -0.1
            Done = False
        elif self.stepCnt > 400:
            # reward = float(-200*200)
            reward = 0
            Done = True
        else :
            try:
                result = self._getObservation(self.pos)
                if not result:
                    reward = -0.01
                    Done = True
                else:
                    Done = False
            except FunctionTimedOut as e:
                print(e)
                reward = 0
                Done = True
                return self.observation, reward, Done ,info

        if not Done:
            self.path.append(copy.copy(self.pos))
            tempGridCnt = countUnkown(self.image)
            exploreReward = (self.unknownGridNum - tempGridCnt + 1) * 0.00005
            self.unknownGridNum = tempGridCnt
            timePunishment = 0
            if(self.observationPolygon.area/self.polygon.area > 0.95):
                finishReward = 1
                # finishReward = 0
                Done = True
            else:
                finishReward = 0
                Done = False
            reward = reward + float(exploreReward+timePunishment+finishReward+obscatlePunishMent)
        return self.observation, reward, Done ,info
