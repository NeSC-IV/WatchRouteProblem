#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from ..Global import pic_size
import numpy as np
import logging
import os
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path/"
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
    iterationNum = 64
    coverageRate = 0.95
    d= 800

    # 随机生成多边形
    for seed in range(3687,20000):
    # for seed in range(20000,0,-1):
        try:
            polygonPints,filename= vis_maps.GetPolygon(seed)
            polygon = shapely.Polygon(polygonPints).buffer(-0.8, cap_style=1, join_style=2).simplify(0.05, preserve_topology=False)
            polygonPints = list(polygon.exterior.coords)
            polygonCoverList, sampleList,order, length, paths, isSuccess = WatchmanRouteProblemSolver(polygon, coverageRate, 32, iterationNum)
            if isSuccess:
                jsonData = {'polygon':polygonPints,'paths':paths}
                print(seed,length)
                with open(dirPath+filename+'.json','w') as f:
                    json.dump(jsonData,f,cls=NpEncoder)
        except:
            pass
            continue

    






