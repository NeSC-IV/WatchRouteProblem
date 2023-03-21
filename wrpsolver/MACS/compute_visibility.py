import shapely
import operator
import math
import numpy as np
from ..Global import *
def GetRayLine(watcher, vertex):
    xGap = vertex[0] - watcher[0]
    yGap = vertex[1] - watcher[1]
    if (xGap == 0):
        extendRate = MyRound(tolerance*zoomRate/abs(yGap), tolerance)
        extendPoint = (watcher[0], watcher[1] + yGap*extendRate)
    elif (yGap == 0):
        extendRate = MyRound(tolerance*zoomRate/abs(xGap), tolerance)
        extendPoint = (watcher[0] + xGap*extendRate, watcher[1])
    else:
        extendRate = max(zoomRate/abs(xGap), zoomRate/abs(yGap))
        extendRate = MyRound(extendRate, tolerance)
        extendPoint = (
            MyRound(watcher[0] + xGap*extendRate, tolerance), MyRound(watcher[1] + yGap*extendRate, tolerance))
    return shapely.LineString([watcher, extendPoint])


def judgeVisible(polygon, watcher, vertex, shortenRate=0.01):
    xGap = vertex.x - watcher.x
    yGap = vertex.y - watcher.y
    # todo 注意多边形的边界

    extendPoint = (MyRound(watcher.x + xGap*(1-shortenRate), tolerance),
                   MyRound(watcher.y + yGap*(1-shortenRate), tolerance))
    if (polygon.covers(shapely.LineString([watcher, extendPoint]))):
        return True

    if xGap != 0:
        extendPoint = (MyRound(watcher.x + xGap*(1-shortenRate), tolerance),
                       MyRound(watcher.y + yGap, tolerance))
        if (polygon.covers(shapely.LineString([watcher, extendPoint]))):
            return True

        extendPoint = (MyRound(watcher.x + xGap*(1+shortenRate), tolerance),
                       MyRound(watcher.y + yGap, tolerance))
        if (polygon.covers(shapely.LineString([watcher, extendPoint]))):
            return True
    if yGap != 0:
        extendPoint = (MyRound(watcher.x + xGap, tolerance),
                       MyRound(watcher.y + yGap*(1-shortenRate), tolerance))
        if (polygon.covers(shapely.LineString([watcher, extendPoint]))):
            return True

        extendPoint = (MyRound(watcher.x + xGap, tolerance),
                       MyRound(watcher.y + yGap*(1-shortenRate), tolerance))
        if (polygon.covers(shapely.LineString([watcher, extendPoint]))):
            return True

    return False


def GetIntersectPointList(intersection):
    pointList = []
    if (type(intersection) == shapely.Point):
        pointList.append(intersection)
    elif (type(intersection) == shapely.MultiPoint):
        pointList = list(intersection.geoms)
    elif (type(intersection) == shapely.GeometryCollection):
        for geometry in list(intersection.geoms):
            if (type(geometry) == shapely.Point):
                pointList.append(geometry)
            elif (type(geometry) == shapely.LineString):
                pointList = pointList+list(geometry.boundary.geoms)
            else:
                print("Unknown geometry type !")
                return None
    return pointList


def GetVisibilityPolygon(polygon, watcher):

    visibilityPolygon = []
    if not polygon.covers(watcher):
        print("The point should be within polygon!")
        return visibilityPolygon
    vertexsList = list(polygon.exterior.coords)
    vertexsList.pop()  # 去除重复点
    vertexsList.reverse()  # 让点按照逆时针排序
    for vertex in vertexsList:



        rayLine = GetRayLine((watcher.x, watcher.y), vertex)
        intersections = (polygon.boundary).intersection(rayLine)
        intersectPointList = GetIntersectPointList(intersections)
        intersectPointList.append(shapely.Point(vertex))

        for intersectPoint in intersectPointList:
            intersectPoint = shapely.Point(
                MyRound(intersectPoint.x, tolerance), MyRound(intersectPoint.y, tolerance))
            if (judgeVisible(polygon, watcher, intersectPoint)):
                visibilityPolygon.append(intersectPoint)

    points = [((MyRound(point.x, tolerance)), (MyRound(point.y, tolerance)))
              for point in visibilityPolygon]
    points = list(dict.fromkeys(points))  # 去重
    points = sorted(points, key=lambda coord: (-135 - math.degrees(math.atan2(*
                                                                              tuple(map(operator.sub, coord, (watcher.x, watcher.y)))[::-1]))) % 360, reverse=False)
    if (type(polygon.boundary) == shapely.MultiLineString):
        return shapely.Polygon(points)
    bndry = (polygon.boundary.coords)
    lineStrings = [shapely.LineString(bndry[k:k+2])
                   for k in range(len(bndry) - 1)]
    lineStrings = [list(ls.coords) for ls in lineStrings]
    i = n = len(points)
    while i <= 2*n:
        p = points[i % n]
        p1 = points[(i+1) % n]
        p2 = points[(i+2) % n]
        if (MyRound((p2[1]-watcher.y)/(p2[0]-watcher.x), 2)) == (MyRound((p1[1]-watcher.y)/(p1[0]-watcher.x), 2)):
            for lineString in lineStrings:
                end = lineString[1]
                start = lineString[0]
                if ((p2[0]-p[0]) == 0) or (end[0]-start[0] == 0):
                    if ((p2[0]-p[0]) == 0) and (end[0]-start[0] == 0):
                        points[(i+1) % n], points[(i+2) %
                                                  n] = points[(i+2) % n], points[(i+1) % n]
                        break
                elif abs(MyRound((p2[1]-p[1])/(p2[0]-p[0]), 2)) == abs(MyRound((end[1]-start[1])/(end[0]-start[0]), 2)):

                    points[(i+1) % n], points[(i+2) %
                                              n] = points[(i+2) % n], points[(i+1) % n]
                    break
        i = i + 1
    return shapely.Polygon(points)