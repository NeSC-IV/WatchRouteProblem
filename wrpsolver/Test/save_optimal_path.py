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
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path_var/"
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
    coverageRate = 0.98
    d = 40

    # 随机生成多边形
    for seed in range(0,30000):
        try:
            pointList,filename, _= vis_maps.GetPolygon(seed)
            polygon = shapely.Polygon(pointList).simplify(0.5,True).buffer(-0.7,join_style=2)
            polygonPints = list(polygon.exterior.coords)
            polygonCoverList, sampleList,order, length, paths, isSuccess = WatchmanRouteProblemSolver(polygon, coverageRate, d, iterationNum)
            if isSuccess:
                jsonData = {'polygon':polygonPints,'paths':paths}
                print(seed,length)
                with open(dirPath+filename+'.json','w') as f:
                    json.dump(jsonData,f,cls=NpEncoder)
            break
        except:
            pass
            continue

    






