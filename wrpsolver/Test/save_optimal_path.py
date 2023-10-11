#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
import logging
import os
import numpy as np
from multiprocessing import Pool,Process
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from ..Global import pic_size
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path_40_3/"
os.makedirs(dirPath, exist_ok=True)
logging.basicConfig(level=logging.INFO)
iterationNum = 64
coverageRate = 0.98
d = 40
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
    for seed in range(begin,end):
        # try:
            pointList,filename,rooms= vis_maps.GetPolygon(seed)
            if(os.path.exists(dirPath+filename+'.json')):
                continue
            polygon = shapely.Polygon(pointList).simplify(0.5,True).buffer(-0.7,join_style=2)
            if(polygon.area > 30000 or polygon.area < 8000 or rooms <= 3 or type(polygon) != shapely.Polygon):
                continue

            polygonPints = list(polygon.exterior.coords)
            polygonCoverList, sampleList,order, length, paths, isSuccess = WatchmanRouteProblemSolver(polygon, coverageRate, d, iterationNum)
            if isSuccess and (length>30):
                jsonData = {'polygon':polygonPints,'paths':paths}
                print(seed,length)
                with open(dirPath+filename+'.json','w') as f:
                    json.dump(jsonData,f,cls=NpEncoder)
            else:
                print(seed,"Convexpolygons: ",len(polygonCoverList))
            
        # except:
        #     print("error")
        #     continue
def SaveOptimalPath():
    step = 100
    seed = step
    while seed <= 35000:
        p = Process(target=SaveSingleOptimalPath,args=(seed-step,seed))
        p.start()
        p.join()
        seed += step

    






