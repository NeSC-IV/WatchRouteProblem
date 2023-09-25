import shapely
import numpy as np
from ..Global import *
from ..Test.draw_pictures import DrawMultiline,DrawPolygon

def getLineList(lines):
    lineList = []
    if (type(lines) == shapely.LineString or type(lines) == shapely.LinearRing):
        lineList.append(lines)
    elif (type(lines) == shapely.MultiLineString):
        for line in lines.geoms:
            lineList.append(line)
    return lineList


def GetSample(polygonList, polygon, dSample):

    import math
    minx, miny, maxx, maxy = polygon.bounds
    maxx = math.ceil(maxx/10)*10
    maxy = math.ceil(maxy/10)*10
    image = np.zeros((int(maxy), int(maxx), 3), dtype=np.uint8)
    DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image, zoomRate=1)

    sampleList = []
    freeSpace = polygon.buffer(-1, join_style=2)
    obstacle = polygon.boundary.buffer(1,join_style=2)
    for polygon in polygonList:
        pointList = []
        lineString = polygon.boundary
        # lineList = getLineList(lineString.difference(freeSpace.boundary.buffer(step)))
        lineList = getLineList(lineString.difference(obstacle))
        for line in lineList:
            DrawMultiline(image,line, color=(200,200,200))
            if line.length < 2:
                continue
            path = dSample
            while (path < line.length):
                point = shapely.line_interpolate_point(line, path)
                if freeSpace.contains(point):
                    pointList.append(point)
                path += dSample
            # end = shapely.line_interpolate_point(line, -1)
            # if freeSpace.contains(end):
            #     pointList.append(end)
            start = shapely.line_interpolate_point(line, 1)
            if freeSpace.contains(start):
                pointList.append(start)
        pointList = list(dict.fromkeys(pointList))  # 去重
        if (len(pointList) > 0):
            sampleList.append(pointList)


        import cv2
        cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/test.png',image)
    return sampleList


def postProcessing(sampleList):
    cityPos = []
    cityGoods = []
    cityClass = []
    n = 0
    for sample in sampleList:
        classify = []
        for point in sample:
            cityPos.append((point.x, point.y))
            cityGoods.append(sampleList.index(sample))
            classify.append(n)
            n += 1
        if (len(classify) > 0):
            cityClass.append(classify)
    for i in range(len(cityPos)):
        x = cityPos[i][0]
        y = cityPos[i][1]
        x = np.round(x).astype(np.int32)
        y = np.round(y).astype(np.int32)
        cityPos[i] = (x,y)
    return ((cityPos, cityGoods, cityClass))


