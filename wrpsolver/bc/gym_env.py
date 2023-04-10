import gym
from gym import spaces
import numpy as np
import cv2
import shapely
import os
from random import choice
import json

from ..Test.draw_pictures import DrawMultiline,DrawPoints,DrawPolygon
from ..MACS.polygons_coverage import FindVisibleRegion
step = 1
grid_size = 200
picDirNames = None
dirPath = os.path.dirname(os.path.abspath(__file__))+"/../../pic_data/"

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
        global picDirNames
        self.polygon = polygon
        if(self.polygon == None):
            if not picDirNames:
                picDirNames = os.listdir(dirPath)
            testJsonDir = dirPath + choice(picDirNames) + '/data.json'
            with open(testJsonDir) as json_file:
                json_data = json.load(json_file)
            self.polygon = shapely.Polygon(json_data['polygon'])
        self.gridPolygon = Polygon2Gird(self.polygon)
        self.observationPolygon = None
        self.pos = None
        self.observation = None
        self.unknownGridNum = None

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
        image = self.observation.reshape(200,200)
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

            self.observationPolygon = visiblePolygon
            self.observation = image.reshape(1,200,200)
        except Exception as e:
            print(e)
            return False
        else:
            return True

    def _get_info(self):
        return {"pos": self.pos}
    def reset(self, startPos=None, seed=None):
        self.observationPolygon = None
        self.pos = None
        self.observation = None
        self.unknownGridNum = None

        if not (startPos):
            gridMap = self.gridPolygon
            x = np.random.randint(0,grid_size)
            y = np.random.randint(0,grid_size)
            while gridMap[0][y][x] == 0:
                x = np.random.randint(0,grid_size)
                y = np.random.randint(0,grid_size)
            startPos = (x,y)
        self.pos = startPos
        self._getObservation(self.pos)
        self.unknownGridNum = 0 
        for grid in np.nditer(self.observation):
            if grid == 150:
                self.unknownGridNum += 1
        info = self._get_info()

        return self.observation
            
    def step(self,action):
        direction = self._action_to_direction[action]
        self.pos += direction
        info = self._get_info()
        # if (self.gridPolygon[0][self.pos[1]][self.pos[0]] == 0):
        #     return None, -200*200 , True, None
        if not self._getObservation(self.pos):
            return self.observation, -200*200 , True, info
        tempGridCnt = 0
        for grid in np.nditer(self.observation):
            if grid == 150:
                tempGridCnt += 1
        exploreReward = self.unknownGridNum - tempGridCnt
        self.unknownGridNum = tempGridCnt
        timePunishment = -10
        if(self.observationPolygon.area/self.polygon.area > 0.9):
            finishReward = 200*200
            Done = True
        else:
            finishReward = 0
            Done = False
        print((exploreReward+timePunishment+finishReward))
        return self.observation, (exploreReward+timePunishment+finishReward), Done ,info
