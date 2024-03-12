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
MAXSTEP = 800
# DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/optimal_path_hole_20_3/complex/"
DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/optimal_path_hole_20_3/"
PIC_DIR_NAMES = os.listdir(DIR_PATH)
IMAGE = np.empty((PIC_SIZE, PIC_SIZE,1), dtype=np.uint8)
IMAGE.fill(150)
LOCAL_RANGE = 64
LOCAL_SHAPE = int((LOCAL_RANGE*2+1)/2)

LOCAL_RANGE1 = 32
LOCAL_SHAPE1 = int((LOCAL_RANGE1*2+1))

LOCAL_RANGE2 = 16
LOCAL_SHAPE2 = int((LOCAL_RANGE2*2+1))

GLOBAL_SHAPE = 64
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
        self.obscatleMap = None
        self.localObs = None
        self.localObs1 = None
        self.localObs2 = None
        self.unknownGridNum = None
        self.frontierList = None
        self.path = []
        self.stepCnt = 0
        self._render = render

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, 1000, shape=(2,), dtype=np.float32),
                # "path": spaces.Box(0, 1000, shape=(10,2), dtype=np.float32),
                "frontier":spaces.Box(0, 1000, shape=(10,), dtype=np.float32),
                "obscatleMap":spaces.Box(0, 255, shape=(16,), dtype=np.float32),
                "localObs":spaces.Box(0, 255, shape=(LOCAL_SHAPE,LOCAL_SHAPE,1), dtype=np.uint8),
                "localObs1":spaces.Box(0, 255, shape=(LOCAL_SHAPE1,LOCAL_SHAPE1,1), dtype=np.uint8),
                # "localObs2":spaces.Box(0, 255, shape=(LOCAL_SHAPE2,LOCAL_SHAPE2,1), dtype=np.uint8),
                # "globalObs":spaces.Box(0, 255, shape=(GLOBAL_SHAPE,GLOBAL_SHAPE,1), dtype=np.uint8),
            }
        )
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
        image = self.initImage.copy()
        point = shapely.Point((pos[0],pos[1]))
        polygon = self.polygon
        visiblePolygon = self.observationPolygon
        result = True
        stepVisiblePolygon = shapely.Point(0,0)

        if self.stepCnt > 0 and self.MoveOutOfRange():
            visiblePolygon = self.observationPolygon
            result = False
        else:
            stepVisiblePolygon = FindVisibleRegion(polygon=polygon,watcher = point, d = 20, useCPP=True)
            if(stepVisiblePolygon == None):
                return False
            if(visiblePolygon == None):
                visiblePolygon = stepVisiblePolygon
            else:
                visiblePolygon = visiblePolygon.union(stepVisiblePolygon)

        if(visiblePolygon == None):
            print("visiblePolygon get failed")
            return False
        # visiblePolygon = SelectMaxPolygon(visiblePolygon).simplify(0.05,True)
        visiblePolygon = SelectMaxPolygon(visiblePolygon).simplify(0.05,True)

        bufferedVisiblePolygon = visiblePolygon.buffer(1,join_style=2)
        if not bufferedVisiblePolygon.is_valid:
            bufferedVisiblePolygon = make_valid(bufferedVisiblePolygon)

        if bufferedVisiblePolygon.intersects(self.o):
            obstacle = (bufferedVisiblePolygon.intersection(self.o)).simplify(0.05,True)
            bufferedObstacle = obstacle.buffer(2,join_style=2)
            frontier = visiblePolygon.boundary.difference(bufferedObstacle)
        else:
            obstacle = None
            frontier = visiblePolygon.boundary


        self.frontierList = GetFrontierList(frontier, self.pos, STEP)
        agent = list([self.stepCnt,self.exploredRange])
        
        # print(obstacle)
        DrawMultiline(image, visiblePolygon, (255))

        # globalObs = image.copy()
        # DrawPoints(globalObs,point.x,point.y,color=(100),size=-1,r=40)
        # DrawMultiline(globalObs,obstacle,color = (0))
        # self.globalObs = cv2.resize(globalObs,(GLOBAL_SHAPE,GLOBAL_SHAPE),interpolation = cv2.INTER_NEAREST)

        DrawMultiline(image,stepVisiblePolygon, (220))

        # DrawMultiline(image,obstacle,color = (0))
        if obstacle == None or not shapely.is_empty(obstacle):
            DrawMultiline(image,obstacle,color = (0))

        # for p in self.path[-PATH_LEN:]:
        for p in self.path[:]:
            x = p[0]
            y = p[1]
            image[y][x] = 80
        self.obscatleMap = self.GetObscatleMap(image,pos[0],pos[1],self.polygon)
        self.localObsImage = self.GetLocalImage(image,pos[0],pos[1],STEP*2)
        DrawMultiline(image,frontier, color=(180))
        DrawPoints(image,point.x,point.y,(30),2)
        for i in range(0, len(self.frontierList), 2):
            goalX = int(math.ceil(self.frontierList[i])*STEP + self.pos[0])
            goalY = int(math.ceil(self.frontierList[i+1])*STEP + self.pos[1])
            # DrawPoints(image,goalX,goalY,color=(10),size=-1,r=2)
        self.localObs = self.GetLocalImage(image,pos[0],pos[1],LOCAL_RANGE)
        self.localObs = cv2.resize(self.localObs,(LOCAL_SHAPE,LOCAL_SHAPE),interpolation = cv2.INTER_NEAREST)
        self.localObs1 = self.GetLocalImage(image,pos[0],pos[1],LOCAL_RANGE1)
        self.localObs2 = self.GetLocalImage(image,pos[0],pos[1],LOCAL_RANGE2)
        self.observationPolygon = visiblePolygon
        self.image = image
        self.observation = {"agent":np.array(agent,dtype = np.float32),
                            "frontier":np.array(self.frontierList,dtype = np.float32),
                            "obscatleMap":np.array(self.obscatleMap),
                            "localObs":self.localObs.reshape(LOCAL_SHAPE,LOCAL_SHAPE,1),
                            "localObs1":self.localObs1.reshape(LOCAL_SHAPE1,LOCAL_SHAPE1,1),
                            }
        
        if self._render:
            savePath = os.path.dirname(os.path.abspath(__file__))+'/../../render_saved/'
            cv2.imwrite(savePath+'tmp0/'+str(self.stepCnt)+'.png',self.image)
            cv2.imwrite(savePath+'tmp1/'+str(self.stepCnt)+'.png',self.localObs)
            cv2.imwrite(savePath+'tmp3/'+str(self.stepCnt)+'.png',self.localObsImage)
            cv2.imwrite(savePath+'tmp2/'+str(self.stepCnt)+'.png',self.localObs1)

        return result
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, seed=None, polygon=None, startPoint = None):
        self.observationPolygon = None
        self.obstacle = None
        self.observation = self.observation_space.sample()
        self.globalObs = None
        self.obscatleMap = None
        self.localObs = None
        self.localObs1 = None
        self.unknownGridNum = None
        self.exploredRange = 0
        self.path = []
        self.stepCnt = 0
        self.polygon = polygon
        self.pos = startPoint
        self.initImage = None
        self.polygonFile = None
        if not self.polygon:
            self.polygon,self.polygonFile = RandomGetPolygon(self._render,seed)
        if not self.pos:
            self.pos = GetStartPoint(self.polygon)
        self.path = [list(self.pos) for _ in range(PATH_LEN)]
        self.polygon = self.polygon.buffer(0.7,join_style=2)
        self.rate = self.polygon.area / (100*100)
        minx, miny, maxx, maxy = self.polygon.bounds
        self.maxx = math.ceil(maxx/10)*10
        self.maxy = math.ceil(maxy/10)*10
        self.initImage = np.zeros((self.maxy, self.maxx, 1), dtype=np.uint8)
        self.initImage.fill(150)
        self.image = self.initImage
        PIC_SIZE = (max(maxx,maxy))+100
        self.o = shapely.Polygon([(0,0),(PIC_SIZE,0),(PIC_SIZE,PIC_SIZE),(0,PIC_SIZE)]).difference(self.polygon)


        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        self.unknownGridNum = CountUnkown(self.image)
        self.exploredRange = self.observationPolygon.area/self.polygon.area
        return self.observation,None
    
    def step(self,action):
        #定义变量
        timePunishment = -1e-2 
        self.rate = 1
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
            tempExploredRange = self.observationPolygon.area/self.polygon.area
            exploreReward = (tempExploredRange - self.exploredRange)
            exploreReward = exploreReward if exploreReward > 1e-4 else 0
            self.exploredRange = tempExploredRange
            
            repeatPunishment = - 0.01 * (self.pos in self.path[-PATH_LEN:])
            if(self.observationPolygon.area/self.polygon.area > 0.98):
                reward+=10
            else:
                Done = False
            reward += float(exploreReward)
            reward *= self.rate
            reward += float(timePunishment+repeatPunishment)
        #更新路径
        self.path.append(copy.copy(self.pos))
        return self.observation, reward*gamma, Done ,False, info
    
    
    def MoveOutOfRange(self):
        oneStep = (self._action_to_direction[self.action]/STEP).astype(np.int32)
        pos = np.array(self.pos)
        for i in range(0,STEP):
            tempPos = (pos - (oneStep * i)).astype(np.int32)
            x = tempPos[0]
            y = tempPos[1]
            p = shapely.Point(x,y)
            #out of boundary
            if (x >= self.maxx) or (y >= self.maxy) or (x <= 0) or (y <= 0):
                return True
            # obstacle
            if (self.image[y][x] == 0)  or (self.image[y][x] == 150) or (not self.polygon.contains(p)):
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
    

    def GetObscatleMap(self, image, posX , posY,polygon = None):
        def getStepObscatle(image, posX, posY, begin, end, polygon):
            result = [0,0,0,0,0,0,0,0] # 0-free 1-obstacle 2-history path
            neibors = [(1, 0),(-1, 0),(0, 1),(0, -1),(1,1),(-1,-1),(-1,1),(1,-1)]
            boundX = image.shape[1]
            boundY = image.shape[0]

            for i,direction in enumerate(neibors):
                dX,dY = direction
                for j in range(begin, end+1):
                    x = posX + dX * j
                    y = posY + dY * j
                    if (x >= boundX) or (y >= boundY) or (x <= 0) or (y <= 0) \
                    or image[y][x] == 0 or image[y][x] == 150: #todo 加入包含判断？
                        result[i] = 1
                        break
                    if polygon and (not polygon.contains(shapely.Point(x,y))):
                        result[i] = 1
                        break

            for i,direction in enumerate(neibors):
                dX,dY = direction
                x = posX + dX * STEP
                y = posY + dY * STEP
                if result[i] == 0 and image[y][x] == 80:
                    result[i] = 2
            # print(result)
            return result


        result = [0] * 16
        result[0:8] = getStepObscatle(image,posX,posY,0,STEP,polygon)
        result[8:16] = getStepObscatle(image,posX,posY,STEP,STEP*2,polygon)#todo 取消第二步观测？
        return result

    def DrawPath(self,image):
        print(self.path)

def RandomGetPolygon(test = False,seed=None):
    if(test):
        jsonFiles = PIC_DIR_NAMES[:int(len(PIC_DIR_NAMES)*0.8)]
        # jsonFiles = PIC_DIR_NAMES[int(len(PIC_DIR_NAMES)*0.8):]

    else:
        jsonFiles = PIC_DIR_NAMES[:int(len(PIC_DIR_NAMES)*0.8)]
    while True:
        fileName = ''
        while '.json' not in fileName:
            fileName = choice(jsonFiles)
        if seed:
            fileName = seed + ".json"
        testJsonDir = DIR_PATH + fileName
        with open(testJsonDir) as json_file:
            jsonData = json.load(json_file)
        points = jsonData['polygon']
        holes = jsonData['hole']
        polygon = shapely.Polygon(shell=points,holes=holes)
        # print(json_data['polygon'])
        # image = IMAGE.copy()
        # DrawMultiline(image,polygon, color=(255))
        # cv2.imwrite('test.png',image)
        if polygon.is_valid and polygon.area < 30000 and polygon.area > 8000:
            return polygon,fileName
        

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


def GetFrontierList(frontier,pos,step):
    ft_list = []
    ft_num = 5
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
        ft_list.extend(0 for _ in range(ft_num*2))
    elif (len(ft_list) < ft_num*2):
        while(len(ft_list) < ft_num*2):
            ft_list.append(ft_list[-2])
            ft_list.append(ft_list[-2])
    else:
        ft_list = ft_list[:ft_num*2]

    ft_list = np.array(ft_list)
    pos_array = np.array(list(pos)*ft_num)
    ft_list = (ft_list - pos_array)/step
    return ft_list

