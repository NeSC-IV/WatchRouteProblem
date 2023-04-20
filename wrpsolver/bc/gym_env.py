import gym
from gym import spaces
import numpy as np
import cv2
import shapely
import os
from random import choice
import json
import copy

from ..Test.draw_pictures import DrawMultiline,DrawSinglePoint,DrawPolygon,DrawPoints
from ..MACS.polygons_coverage import FindVisibleRegion
step = 1
grid_size = 200
picDirNames = None
dirPath = os.path.dirname(os.path.abspath(__file__))+"/../../pic_data/"
def countUnkown(image):
    ret,thresh=cv2.threshold(image,254,255,cv2.THRESH_BINARY_INV)
    cnt = cv2.countNonZero(thresh)
    return cnt
def Polygon2Gird(polygon):

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    points = list(polygon.exterior.coords)
    points = np.array(points)
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        grid = cv2.fillPoly(grid, [points], 255)
    else:
        grid = cv2.fillPoly(grid, points, 255)

    return grid.reshape(1,200,200)
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, polygon=None, startPos=None):
        self.polygon = polygon
        self.startPos = startPos
        self.pos = startPos
        self.polygonInited = True if polygon is not None else False
        self.posInited = True if startPos is not None else False
        self.gridPolygon = None
        self.observationPolygon = None
        self.observation = None
        self.unknownGridNum = None
        self.path = []
        self.stepCnt = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,grid_size, grid_size), dtype=np.uint8)

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

    def _getObservation(self,pos):
        # 更新observationPolygon和observation

        self.observation = np.empty((1, grid_size, grid_size), dtype=np.uint8)
        self.observation.fill(150)
        point = shapely.Point(pos)
        polygon = self.polygon
        image = self.observation[0]
        visiblePolygon = self.observationPolygon

        try:
            if(visiblePolygon == None):
                visiblePolygon = FindVisibleRegion(self.polygon,point,800)
            else:
                visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon,point,800))
            unknownRegion = visiblePolygon.boundary.difference(polygon.boundary.buffer(1))
            obcastle = visiblePolygon.boundary.difference(unknownRegion.buffer(1))

            DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)
            DrawMultiline(image,unknownRegion,(150))
            DrawMultiline(image,obcastle,color = (0))
            DrawPoints(image,point.x,point.y,(30))
            for point in self.path:
                x = point[0]
                y = point[1]
                image[y][x] = 50

            self.observationPolygon = visiblePolygon
            self.observation[0] = image
        except Exception as e:
            print(e)
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
            if not picDirNames:
                picDirNames = os.listdir(dirPath)
            testJsonDir = dirPath + choice(picDirNames) + '/data.json'
            with open(testJsonDir) as json_file:
                json_data = json.load(json_file)
            self.polygon = shapely.Polygon(json_data['polygon'])

        self.gridPolygon = Polygon2Gird(self.polygon)

        if (startPos):
            self.pos = startPos
        elif not self.posInited:
            gridMap = self.gridPolygon
            x = np.random.randint(0,grid_size)
            y = np.random.randint(0,grid_size)
            while gridMap[0][y][x] == 0:
                x = np.random.randint(0,grid_size)
                y = np.random.randint(0,grid_size)
            startPos = (x,y)
            self.pos = startPos
        else:
            self.pos = self.startPos
        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        self.unknownGridNum = countUnkown(self.observation[0])
        info = self._get_info()

        return self.observation
            
    def step(self,action):
        direction = self._action_to_direction[action]
        self.pos += direction
        info = self._get_info()
        self.stepCnt += 1
        if abs(self.pos[1])>=200 or abs(self.pos[0])>=200 or (self.observation[0][self.pos[1]][self.pos[0]] == 0):
            reward = float(-200*200)
            Done = True
        elif self.stepCnt > 400:
            # reward = float(-200*200)
            reward = 0
            Done = True
        elif not self._getObservation(self.pos):
            reward = float(-200*200)
            Done = True
        else:
            self.path.append(copy.copy(self.pos))
            tempGridCnt = countUnkown(self.observation[0])
            exploreReward = self.unknownGridNum - tempGridCnt
            self.unknownGridNum = tempGridCnt
            timePunishment = -10
            if(self.observationPolygon.area/self.polygon.area > 0.9):
                # finishReward = 200*200
                finishReward = 0
                Done = True
            else:
                finishReward = 0
                Done = False
            reward = float(exploreReward+timePunishment+finishReward)
        return self.observation, reward, Done ,info
