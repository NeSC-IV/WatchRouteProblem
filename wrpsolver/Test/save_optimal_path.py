#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
import logging
import os
import sys
import numpy as np
from multiprocessing import Pool,Process
from func_timeout import func_timeout, FunctionTimedOut
from . import vis_maps
from . import hole_maps
from ..WRP_solver import WatchmanRouteProblemSolver

dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path_20_3/"
os.makedirs(dirPath, exist_ok=True)
logging.basicConfig(level=logging.INFO)
iterationNum = 64
coverageRate = 0.97
d = 20
step = 3
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def SaveSingleOptimalPath(begin,end):
    useHoleMap = False
    for seed in range(begin,end):

        if useHoleMap:
            polygon = hole_maps.GetPolygon(seed)
            savePath = dirPath+str(seed)+'.json'
        else:
            pointList,filename,rooms= vis_maps.GetPolygon(seed)
            savePath = dirPath+filename+'.json'
            polygon = shapely.Polygon(pointList).simplify(0.5,True).buffer(-0.7,join_style=2)
            if(polygon.area > 30000 or polygon.area < 5000 or rooms <= 3 or type(polygon.buffer(-2, join_style=2)) != shapely.Polygon):
                continue
        if(type(polygon) != shapely.Polygon) or os.path.exists(savePath):
            continue
        polygonCoverList, sampleList,order, length, paths, isSuccess = WatchmanRouteProblemSolver(polygon, coverageRate, d, iterationNum,step=step)

        
        polygonPoints = list(polygon.exterior.coords)
        polygonHoles = [list(interior.coords) for interior in polygon.interiors]
        if isSuccess and (length>30):
            jsonData = {'polygon':polygonPoints,'hole':polygonHoles,'paths':paths}  #'   paths'
            print(seed,"succeed",length)
            with open(savePath,'w') as f:
                json.dump(jsonData,f,cls=NpEncoder)
                f.close()
        else:
            print(seed,"failed"," length:",length," isSuccess:",isSuccess)

def SaveOptimalPath(begin = 0,end = 10000):

    SaveSingleOptimalPath(begin,end)

    # step = 100
    # seed = step
    # while seed < 10000:
    #     p = Process(target=SaveSingleOptimalPath,args=(seed-step,seed))
    #     p.start()
    #     p.join()
    #     seed += step

    






