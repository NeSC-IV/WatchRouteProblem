#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from ..Global import zoomRate
import numpy as np
import logging
import os
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path_picsize100/"
os.makedirs(dirPath, exist_ok=True)
logging.basicConfig(level=logging.INFO) 
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def SaveOptimalPath():
    iterationNum = 24
    coverageRate = 0.95
    d= 800

    # 随机生成多边形
    # for seed in range(19987,10000,-1):
    for seed in range(0,10000,1):
        try:
            polygonPints,filename= vis_maps.GetPolygon(seed)
            polygon = shapely.Polygon(polygonPints).buffer(-7).simplify(0.5,False)
            polygonPints = list(polygon.exterior.coords)
            polygonCoverList, sampleList,order, length, paths = WatchmanRouteProblemSolver(polygon, coverageRate, iterationNum,d)
            # jsonData = {'polygon':polygonPints.tolist(),'paths':paths}
            jsonData = {'polygon':polygonPints,'paths':paths}
            logging.info(seed)
            with open(dirPath+filename+'.json','w') as f:
                json.dump(jsonData,f,cls=NpEncoder)
        except:
            pass
            continue

    






