#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import time
import json
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from ..MACS.polygons_coverage import FindVisibleRegion
from .draw_pictures import *

def getVertex(polygon):
    vertexList = []
    if(type(polygon) == shapely.Polygon):
        vertexList = list(polygon.exterior.coords)
    elif(type(polygon) == shapely.LineString):
        vertexList = list(polygon.coords)
    elif(type(polygon == shapely.MultiLineString)):
        for line in list(polygon.geoms):
            vertexList += list(line.coords)
    return vertexList
def GetTrajectory(seed = 1):
    iterationNum = 10
    coverageRate = 0.98

    # 随机生成多边形
    polygon = shapely.Polygon(vis_maps.GetPolygon(seed))

    polygonCoverList, sampleList,order, length, paths = WatchmanRouteProblemSolver(
        polygon, coverageRate, iterationNum,300)
    
    time1 = time.time()
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)


    step = int(zoomRate/50)
    actionDict = {(step,0):0,(-step,0):1,(0,step):2,(0,-step):3,
                  (step,step):4,(-step,-step):5,(-step,step):6,(step,-step):7}

    visibleArray = []
    unknownArray = []
    obcastleArray = []
    actionArray = []
    visiblePolygon = None
    for path in paths:
        # DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)
        # DrawPath(image,path)
        for i in range(len(path)-1):
            point = shapely.Point(path[i])
            pointNext = shapely.Point(path[i+1])
            if(visiblePolygon == None):
               visiblePolygon = FindVisibleRegion(polygon,point,3000)
            else:
                visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon,point,3000))
            unknownRegion = visiblePolygon.boundary.difference(polygon.boundary.buffer(zoomRate/500))
            obcastle = visiblePolygon.boundary.difference(unknownRegion)
            visibleArray.append(getVertex(visiblePolygon))
            unknownArray.append(getVertex(unknownRegion))
            obcastleArray.append(getVertex(obcastle))
            actionArray.append(actionDict[int(pointNext.x-point.x),int(pointNext.y-point.y)])

        #     DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)
        #     DrawPolygon( list(visiblePolygon.exterior.coords), (0, 255, 0), image)
        #     DrawPoints(image, point.x, point.y)
        # cv2.imshow('polygons', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    jsonData = {"visibleArray":visibleArray,"unknownArray":unknownArray,"obcastleArray":obcastleArray,"actionArray":actionArray}
    with open('data.json','w') as f:
        json.dump(jsonData,f)
    time2 = time.time()
    print(time2-time1)






