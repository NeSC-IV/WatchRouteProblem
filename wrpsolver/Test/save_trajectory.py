#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import json
import os
import shutil
from multiprocessing import Pool
from ..MACS.polygons_coverage import FindVisibleRegion,SelectMaxPolygon
from .draw_pictures import *
from ..Global import step
from shapely.validation import make_valid
IMAGE = np.empty((100, 100,1), dtype=np.uint8)
IMAGE.fill(150)
dirPath = os.path.dirname(os.path.abspath(__file__))+"/optimal_path_picsize100_new/"
savePath = './wrpsolver/Test/pic_data_picsize100_pos/'
os.makedirs(savePath, exist_ok=True)

def GetpathIDs(dirPath):

    pathIDs = os.listdir(dirPath)
    pathIDs = [pathID.split('.')[0] for pathID in pathIDs]
    return pathIDs


def GetLocalImage(image,x,y):
    _range = 10
    y_low = max((y-_range),0)
    y_high = min((y+_range),pic_size)
    x_low = max((x-_range),0)
    x_high = min((x+_range),pic_size)

    newImage = image.copy()
    ret,newImage=cv2.threshold(newImage,31,255,cv2.THRESH_TRUNC)
    DrawPoints(newImage,x,y,(255),-1)
    newImage = newImage[y_low:y_high,[row for row in range(x_low,x_high)]]
    newImage = cv2.resize(newImage,(pic_size,pic_size),interpolation = cv2.INTER_NEAREST)
    return newImage


def GetSingleTrajectory(pathID):
        if os.path.exists(savePath + pathID):
            pass
        else:
            os.mkdir(savePath + pathID )
        d = 32
        step = 1
        image = IMAGE.copy()
        with open(dirPath + pathID+'.json') as jsonFile:
            jsonData = json.load(jsonFile)
            pointList = jsonData['polygon']
            polygon = shapely.Polygon(pointList)
            o = shapely.Polygon([(0,0),(100,0),(100,100),(0,100)]).difference(polygon).buffer(-0.5)
            DrawMultiline(image,polygon,color = (0))
            cv2.imwrite(savePath + pathID + '/' + 'polygon' + '.png',image)
            paths = jsonData['paths']
            if(len(paths) < 3):
                shutil.rmtree(savePath + pathID)
                return


        actionDict = {(step,0):0,(-step,0):1,(0,step):2,(0,-step):3,
                    (step,step):4,(-step,-step):5,(-step,step):6,(step,-step):7}

        actionArray = []
        visiblePolygon = None
        cnt = 0
        for i in range(len(paths)):
            path = paths[i]
            for j in range(len(path)-1):
                
                image = IMAGE.copy()
                image_ft = IMAGE.copy()
                image_ft.fill(255)
                point = shapely.Point(path[j])
                if(visiblePolygon == None):
                    visiblePolygon = FindVisibleRegion(polygon,point,d,True)
                else:
                    visiblePolygon = SelectMaxPolygon(visiblePolygon.union(FindVisibleRegion(polygon,point,d,True)))
                if(visiblePolygon == None):
                    print("visiblePolygon get failed")
                    return
                visiblePolygon = visiblePolygon.simplify(0.1,False)
                obcastle = visiblePolygon.buffer(2).intersection(o)
                frontier = visiblePolygon.boundary.difference(obcastle.buffer(4)).simplify(0.05,False)


                pointNext = shapely.Point(path[j+1])
                actionArray.append(actionDict[round((pointNext.x-point.x)),round((pointNext.y-point.y))])
                # 

                DrawMultiline(image,obcastle,color = (0))
                DrawMultiline( image,visiblePolygon,color = (255))
                DrawMultiline(image,frontier, color=(200))
                DrawPoints(image,point.x,point.y,(30),-1)

                DrawMultiline(image_ft,frontier, color=(150))
                DrawPoints(image_ft,point.x,point.y,(30),-1)
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
                localImage = GetLocalImage(image,path[j][0],path[j][1])
                cv2.imwrite(savePath + pathID + '/' + str(cnt) + '.png',image)
                cv2.imwrite(savePath + pathID + '/' + str(cnt) + '_pos.png',localImage)
                cv2.imwrite(savePath + pathID + '/' + str(cnt) + '_ft.png',image_ft)
                cnt+=1

        jsonData1 = {"actionArray":actionArray}
        jsonData = {**jsonData1,**jsonData}
        if(visiblePolygon.area / polygon.area < 0.8):
            shutil.rmtree(savePath + pathID)
            print("visiblePolygon.area / polygon.area < 0.8")
            return
        with open(savePath + pathID + '/' + 'data.json','w') as f:
            json.dump(jsonData,f)
def GetTrajectory(seed = 1):


    # 随机生成多边形

    pool = Pool(threadNum)
    pathIDs = GetpathIDs(dirPath)
    pool.map(GetSingleTrajectory,pathIDs[:])
    pool.close()
    pool.join()







