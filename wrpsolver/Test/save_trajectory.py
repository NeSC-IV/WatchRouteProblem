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
from shapely.validation import make_valid
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path_picsize100/"
savePath = './wrpsolver/Test/pic_data_picsize100/'
os.makedirs(savePath, exist_ok=True)

def GetpathIDs(dirPath):

    pathIDs = os.listdir(dirPath)
    pathIDs = [pathID.split('.')[0] for pathID in pathIDs]
    return pathIDs


def GetSingleTrajectory(pathID):
        if not os.path.exists(savePath + pathID):
            os.mkdir(savePath + pathID )
        try:
            d = 800
            step = 1
            image = np.zeros((pic_size, pic_size, 1), dtype=np.uint8)


            with open(dirPath + pathID+'.json') as jsonFile:
                jsonData = json.load(jsonFile)
                polygon = shapely.Polygon(jsonData['polygon'])
                a = shapely.Polygon([(0,0),(zoomRate,0),(zoomRate,zoomRate),((0,zoomRate))])
                # a = polygon.buffer(1)
                o = a.difference(polygon)
                paths = jsonData['paths']
                if(len(paths) < 3):
                    shutil.rmtree(savePath + pathID)
                    return


            actionDict = {(step,0):0,(-step,0):1,(0,step):2,(0,-step):3,
                        (step,step):4,(-step,-step):5,(-step,step):6,(step,-step):7,(0,0):8}

            actionArray = []
            visiblePolygon = None
            cnt = 0
            for i in range(len(paths)):
                path = paths[i]
                for j in range(len(path)-1):
                    
                    point = shapely.Point(path[j][0]*10,path[j][1]*10)
                    if(visiblePolygon == None):
                        visiblePolygon = FindVisibleRegion(polygon,point,d,True)
                    else:
                        visiblePolygon = visiblePolygon.union(FindVisibleRegion(polygon,point,d,True))
                    # obcastle = visiblePolygon.intersection(polygon.boundary.buffer(0.5))
                    obcastle = o.intersection(visiblePolygon.buffer(zoomRate/pic_size)).buffer(0.5)

                    pointNext = shapely.Point(path[j+1][0]*10,path[j+1][1]*10)
                    actionArray.append(actionDict[round((pointNext.x-point.x)/10),round((pointNext.y-point.y)/10)])
                    image.fill(150)

                    DrawMultiline(image,obcastle,color = (0),zoomRate=(pic_size/zoomRate))
                    DrawPolygon( list(visiblePolygon.exterior.coords), (255), image,zoomRate=(pic_size/zoomRate))
                    DrawPoints(image,point.x,point.y,(30),zoomRate=(pic_size/zoomRate))
                    # DrawSinglePoint(image,point.x,point.y,(30))
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
            if(visiblePolygon.area / polygon.area < 0.9):
                shutil.rmtree(savePath + pathID)
                return
            with open(savePath + pathID + '/' + 'data.json','w') as f:
                json.dump(jsonData,f)
        except Exception as e:
            print(e)
            shutil.rmtree(savePath + pathID)
def GetTrajectory(seed = 1):


    # 随机生成多边形

    pool = Pool(threadNum)
    pathIDs = GetpathIDs(dirPath)
    pool.map(GetSingleTrajectory,pathIDs[:])
    pool.close()
    pool.join()







