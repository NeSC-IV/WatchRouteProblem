#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import time
import json
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from ..MACS.polygons_coverage import FindVisibleRegion
from .draw_pictures import *
from ..Global import step

def drawMultiline(image, multiLine,color = (0, 25, 255)):
    
    def drawSingleline(image,line,color = (0, 25, 255)):
        pointList = list(line.coords)
        length = len(pointList)
        for i in range(length-1):
            DrawLine(image,pointList[i],pointList[i+1],color)

    if(type(multiLine) == shapely.LineString):
        drawSingleline(image,multiLine,color)
    elif(type(multiLine) == shapely.MultiLineString):
        for line in list(multiLine.geoms):
            drawSingleline(image,line,color)
    else:
        print("unknown type")


def GetTrajectory(seed = 1):
    iterationNum = 10
    coverageRate = 0.95
    d= 800
    image = np.empty((pic_size, pic_size, 1), dtype=np.uint8)
    image.fill(150)
    # 随机生成多边形
    pointList,filename = vis_maps.GetPolygon(seed)
    polygon = shapely.Polygon(pointList)

    polygonCoverList, sampleList,order, length, paths = WatchmanRouteProblemSolver(
        polygon, coverageRate, iterationNum,d)
    polintList = pointList.tolist()
    polygon = shapely.Polygon(pointList)
    time1 = time.time()


    # DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)
    step = 1
    actionDict = {(step,0):0,(-step,0):1,(0,step):2,(0,-step):3,
                  (step,step):4,(-step,-step):5,(-step,step):6,(step,-step):7}


    actionArray = []
    posArray=[]
    visiblePolygon = None
    cnt = 0
    for path in paths:
        for j in range(len(path)-1):
            point = shapely.Point(path[j])
            if(visiblePolygon == None):
               visiblePolygon = FindVisibleRegion(polygon,point,d)
            else:
                visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon,point,d))
            unknownRegion = visiblePolygon.boundary.difference(polygon.boundary.buffer(zoomRate/500))
            obcastle = visiblePolygon.boundary.difference(unknownRegion.buffer(zoomRate/500))

            pointNext = shapely.Point(path[j+1])
            actionArray.append(actionDict[round(pointNext.x-point.x),round(pointNext.y-point.y)])
            posArray.append((point.x,point.y))
            DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)
            drawMultiline(image,unknownRegion,(150))
            drawMultiline(image,obcastle,color = (0))
            DrawPoints(image,point.x,point.y,(30))

            cv2.imwrite('./pic_data/'+str(cnt)+'.png',image)
            cnt+=1

    # cv2.imshow('polygons', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    jsonData = {"actionArray":actionArray}
    with open('data.json','w') as f:
        json.dump(jsonData,f)
    time2 = time.time()
    print(time2-time1)






