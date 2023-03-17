import shapely
from Global import *


def getLineList(lines):
    lineList = []
    if (type(lines) == shapely.LineString):
        lineList.append(lines)
    elif (type(lines) == shapely.MultiLineString):
        for line in lines.geoms:
            lineList.append(line)
    return lineList


def GetSample(polygonList, freeSpace, dSample, image):
    sampleList = []
    for polygon in polygonList:
        pointList = []
        lineString = polygon.boundary
        lineList = getLineList(lineString.difference(
            freeSpace.boundary.buffer(zoomRate/1000)))
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
    return ((cityPos, cityGoods, cityClass))


