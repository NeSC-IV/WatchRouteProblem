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
RANGE = 32
LOCAL_SHAPE = (RANGE*2+1)*2
DIR_PATH = os.path.dirname(os.path.abspath(__file__))+"/../Test/pic_data_picsize100_new/"
IMAGE = np.zeros((PIC_SIZE, PIC_SIZE,1), dtype=np.uint8)
class GridWorldEnv(gym.Env):

    def __init__(self, polygon=None, startPos=None, seed = None, render = False):
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

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, PIC_SIZE, shape=(6,), dtype=int),
                "localImage":spaces.Box(0, 255, shape=(3,3), dtype=np.uint8),
                "localImage1":spaces.Box(0, 255, shape=(LOCAL_SHAPE,LOCAL_SHAPE,1), dtype=np.uint8),
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
        image = self.image.copy()

        try:
            if self.MoveOutOfRange(image):
                return False
            for p in self.path:
                x = p[0]
                y = p[1]
                image[y][x] = 80
            localImage = GetLocalImage(image,pos[0],pos[1])
            DrawSinglePoint(image,self.goal[0],self.goal[1],(150),2)
            localImage1 = GetLocalImage(image,pos[0],pos[1],_range = RANGE)
            localImage1 = cv2.resize(localImage1,(LOCAL_SHAPE,LOCAL_SHAPE),interpolation = cv2.INTER_NEAREST)
            DrawSinglePoint(image,self.pos[0],self.pos[1],(30),2)
            agent = []
            for p in self.path[-1:]:
                agent += p 
            agent = agent + self.pos + self.goal
            self.observation = {"agent":np.array(agent),"localImage":localImage,"localImage1":localImage1.reshape(LOCAL_SHAPE,LOCAL_SHAPE,1)}

            if self.render:
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp0/'+str(self.stepCnt)+'.png',image)
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/'+str(self.stepCnt)+'.png',localImage1)
                cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/'+str(self.stepCnt)+'.png',localImage)
        except Exception as e:
            print(e)
            return False
        else:
            return True
        
    def _get_info(self):
        return {"pos": self.pos}
    
    def reset(self, seed=None):
        self.observation = self.observation_space.sample()
        self.path = [[0,0] for _ in range(1)]
        self.stepCnt = 0

        if not self.polygonInited:
            self.polygon = RandomGetPolygon().simplify(0.05, preserve_topology=True)
        if not self.startPosInited:
            self.pos = GetStartPoint(self.polygon)
        self.image = IMAGE.copy()

        DrawMultiline(self.image,self.polygon, color=(255))
        self.goal = GetStartPoint(self.polygon)


        self._getObservation(self.pos)
        self.path.append(copy.copy(self.pos))
        self.distance = math.hypot(self.pos[0]-self.goal[0], self.pos[1]-self.goal[1])

        return self.observation,None
    
    def step(self,action):

        #定义变量
        timePunishment = 0
        reward = 0
        info = self._get_info()
        gamma = 2
        Done = True
        distance = math.hypot((self.pos[0]-self.goal[0]), (self.pos[1]-self.goal[1]))
        #更新位置
        direction = self._action_to_direction[action]
        self.pos = (self.pos+direction).tolist()
        self.stepCnt += 1

        #agent步数是否到达上限
        if self.stepCnt >= MAXSTEP:
            pass

        elif not self._getObservation(self.pos):
            reward -= 1

        else:
            Done = False

        #计算奖励
        if not Done:
            repeatPunishment = - 0.01 if (self.pos in self.path) else 0.001
            distanceReward = (self.distance - distance) * 0.005
            if distance < 4:
                reward += 1
                Done = True
            reward += float(distanceReward+timePunishment+repeatPunishment)
            self.distance = distance
        #更新路径
        self.path.append(copy.copy(self.pos))
        return self.observation, reward*gamma, Done ,False,info
    
    def MoveOutOfRange(self,image):
        x = self.pos[0]
        y = self.pos[1]
        #out of boundary
        if (x >= PIC_SIZE) or (y >= PIC_SIZE) or (x <= 0) or (y <= 0):
            return True
        if (image[y][x] == 0):
            return True
        return False

def RandomGetPolygon():
    global PIC_DIR_NAMES
    while True:        
        # if not PIC_DIR_NAMES:
        #     PIC_DIR_NAMES = os.listdir(DIR_PATH)
        # testJsonDir = DIR_PATH + choice(PIC_DIR_NAMES) + '/data.json'
        # with open(testJsonDir) as json_file:
        #     json_data = json.load(json_file)
        # polygon = shapely.Polygon(json_data['polygon'])
        polygon = GetPolygon(random.randint(0,35000))
        if polygon.is_valid:
            return polygon

def GetPolygon(seed):
    json_path = os.path.dirname(os.path.abspath(__file__))+"/../Test/json/"
    map_file = os.path.dirname(os.path.abspath(__file__))+"/../Test/map_id_35000.txt"
    map_ids = np.loadtxt(map_file, str)

    file_name = map_ids[seed]
    # print(file_name)
    with open(json_path + '/' + file_name + '.json') as json_file:
        json_data = json.load(json_file)


    bbox = json_data['bbox']
    maxNum = max(bbox['max'][0],bbox['max'][1])
    verts = (np.array(json_data['verts']) * PIC_SIZE / math.ceil(maxNum)).astype(int)
    return shapely.Polygon(verts)
     
def GetStartPoint(polygon):
    temppolygon = polygon.buffer(-STEP)
    minx, miny, maxx, maxy = temppolygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if temppolygon.covers(p):
            return [int(p.x),int(p.y)]
    
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
    # ret,newImage=cv2.threshold(newImage,81,255,2)
    # DrawPoints(newImage,x,y,(255),-1)
    newImage = newImage[y_low:y_high+1,[row for row in range(x_low,x_high+1)]]
    # newImage = cv2.resize(newImage,(paddleSize,paddleSize),interpolation = cv2.INTER_NEAREST)
    return newImage

def getDirection(P1, P2, img):
    x1 = P1[0]
    y1 = P1[1]
    x2 = P2[0]
    y2 = P2[1]
    image = IMAGE.copy()
    cv2.line(image,(x1,y1),(x2,y2),255,1)
    coords = np.argwhere(image)
    for pos in coords:
        if(img[pos[0]][pos[1]] == 255):
            img[pos[0]][pos[1]] = 200