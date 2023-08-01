import shapely
import numpy as np
from ..Global import *


def getLineList(lines):
    lineList = []
    if (type(lines) == shapely.LineString):
        lineList.append(lines)
    elif (type(lines) == shapely.MultiLineString):
        for line in lines.geoms:
            lineList.append(line)
    return lineList


def GetSample(polygonList, freeSpace, dSample):
    sampleList = []
    for polygon in polygonList:
        pointList = []
        lineString = polygon.boundary
        lineList = getLineList(lineString.difference(
            freeSpace.boundary.buffer(step/2)))
        for line in lineList:
            path = 0
            while (path < line.length):
                point = shapely.line_interpolate_point(line, path)
                if freeSpace.covers(point):
                    pointList.append(point)
                path += dSample
            end = shapely.get_point(line, -1)
            if freeSpace.covers(end):
                pointList.append(end)
            start = shapely.get_point(line, 0)
            if freeSpace.covers(start):
                pointList.append(start)
        pointList = list(dict.fromkeys(pointList))  # 去重
        sampleList.append(pointList)
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


