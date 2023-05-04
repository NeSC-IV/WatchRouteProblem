#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
import os
import shutil
from multiprocessing import Pool
from ..MACS.polygons_coverage import FindVisibleRegion
from .draw_pictures import *
from ..Global import step
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path/"
savePath = './wrpsolver/Test/pic_data_resize/'
os.makedirs(savePath, exist_ok=True)
def DrawMultiline(image, multiLine,color = (0, 25, 255)):
    
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

def GetpathIDs(dirPath):

    pathIDs = os.listdir(dirPath)
    pathIDs = [pathID.split('.')[0] for pathID in pathIDs]
    return pathIDs


def GetSingleTrajectory(pathID):
        if not os.path.exists(savePath + pathID):
            os.mkdir(savePath + pathID )
        try:
            d = 800
            image = np.empty((pic_size, pic_size, 1), dtype=np.uint8)
            image.fill(150)
            with open(dirPath + pathID+'.json') as jsonFile:
                jsonData = json.load(jsonFile)
                polygon = shapely.Polygon(jsonData['polygon'])
                polygon = polygon
                paths = jsonData['paths']


            actionDict = {(step,0):0,(-step,0):1,(0,step):2,(0,-step):3,
                        (step,step):4,(-step,-step):5,(-step,step):6,(step,-step):7}

            actionArray = []
            visiblePolygon = None
            cnt = 0
            for i in range(len(paths)):
                path = paths[i]
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
                    DrawPolygon( list(visiblePolygon.exterior.coords), (255), image)
                    DrawMultiline(image,unknownRegion,(150))
                    DrawMultiline(image,obcastle,color = (0))
                    # DrawPoints(image,point.x,point.y,(30))
                    DrawSinglePoint(image,point.x,point.y,(30))
                    for k in range(i):
                        prePath = paths[k]
                        for l in range(len(prePath)-1):
                            x = prePath[l][0]
                            y = prePath[l][1]
                            image[y][x] = 80
                    for l in range(j):
                            x = path[l][0]
                            y = path[l][1]
                            image[y][x] = 80
                    cv2.imwrite(savePath + pathID + '/' + str(cnt) + '.png',image)
                    cnt+=1

            jsonData1 = {"actionArray":actionArray}
            jsonData = {**jsonData1,**jsonData}
            with open(savePath + pathID + '/' + 'data.json','w') as f:
                json.dump(jsonData,f)
        except Exception as e:
            print(e)
            shutil.rmtree(savePath + pathID)
def GetTrajectory(seed = 1):


    # 随机生成多边形

    pool = Pool(threadNum)
    pathIDs = GetpathIDs(dirPath)
    pool.map(GetSingleTrajectory,pathIDs[:100])
    pool.close()
    pool.join()







