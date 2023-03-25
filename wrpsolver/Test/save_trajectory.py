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
    coverageRate = 0.99
    d= 8000
    # 随机生成多边形
    polygon = shapely.Polygon(vis_maps.GetPolygon(seed))

    polygonCoverList, sampleList,order, length, paths = WatchmanRouteProblemSolver(
        polygon, coverageRate, iterationNum,d)
    
    time1 = time.time()
    image = np.empty((pic_size, pic_size, 1), dtype=np.uint8)
    image.fill(50)

    # DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)


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
        for i in range(len(path)):
            point = shapely.Point(path[i])
            # pointNext = shapely.Point(path[i+1])
            if(visiblePolygon == None):
               visiblePolygon = FindVisibleRegion(polygon,point,d)
            else:
                visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon,point,d))
            unknownRegion = visiblePolygon.boundary.difference(polygon.boundary.buffer(zoomRate/500))
            obcastle = visiblePolygon.boundary.difference(unknownRegion.buffer(zoomRate/500))

            # visibleArray.append(getVertex(visiblePolygon))
            # obcastleArray.append(getVertex(obcastle))
            # unknownArray.append(getVertex(unknownRegion))
            # actionArray.append(actionDict[round(pointNext.x-point.x),round(pointNext.y-point.y)])


            # DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)
            DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)
            drawMultiline(image,unknownRegion,(150))
            drawMultiline(image,obcastle,color = (0))
            for x in range(pic_size):
                for y in range(pic_size):
                    color = int(image[x][y][0])
                    if(color==255):
                        # visibleArray.append((x,y))
                        pass
                    elif(color==0):
                        obcastleArray.append((x,y))
                    elif(color==150):
                        unknownArray.append((x,y))
            # DrawPoints(image, point.x, point.y,(100,100,100))
        # point = shapely.Point(path[-1])
        # visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon,point,300))
        # DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)

        
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    jsonData = {"visibleArray":visibleArray,"unknownArray":unknownArray,"obcastleArray":obcastleArray,"actionArray":actionArray}
    with open('data.json','w') as f:
        json.dump(jsonData,f)
    time2 = time.time()
    print(time2-time1)






