#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver


def SaveOptimalPath():
    iterationNum = 10
    coverageRate = 0.95
    d= 800
    # 随机生成多边形
    for seed in range(10000):
        try:
            polygonPints,filename= vis_maps.GetPolygon(seed)
            polygon = shapely.Polygon(polygonPints)
            polygonCoverList, sampleList,order, length, paths = WatchmanRouteProblemSolver(polygon, coverageRate, iterationNum,d)
            jsonData = {'polygon':polygonPints.tolist(),'paths':paths}
            with open('optimal_path/'+filename+'.json','w') as f:
                json.dump(jsonData,f)
            break
        except:
            pass
            continue

    






