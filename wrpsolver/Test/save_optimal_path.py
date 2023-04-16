#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
import numpy as np
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
    for seed in range(20000,0,-1):
        try:
            polygonPints,filename= vis_maps.GetPolygon(seed)
            polygon = shapely.Polygon(polygonPints)
            polygonCoverList, sampleList,order, length, paths = WatchmanRouteProblemSolver(polygon, coverageRate, iterationNum,d)
            jsonData = {'polygon':polygonPints.tolist(),'paths':paths}
            with open('wrpsolver/Test/optimal_path/'+filename+'.json','w') as f:
                json.dump(jsonData,f,cls=NpEncoder)
        except:
            pass
            continue

    






