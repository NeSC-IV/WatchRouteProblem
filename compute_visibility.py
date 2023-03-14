import shapely
import random
import random_polygons_generate
import operator
import math
from draw_pictures import *


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
    if not polygon.contains(watcher):
        print("The point should be within polygon!")
        return visibilityPolygon
    vertexsList = list(polygon.exterior.coords)
    vertexsList.pop()  # 去除重复点
    vertexsList.reverse()  # 让点按照逆时针排序
    for vertex in vertexsList:

        image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
        DrawPolygon((pic_size, pic_size, 3), list(
            polygon.exterior.coords), (255, 255, 255), image)
        image = DrawPoints(image, watcher.x, watcher.y, (0, 255, 0))

        rayLine = GetRayLine((watcher.x, watcher.y), vertex)
        intersections = (polygon.boundary).intersection(rayLine)
        intersectPointList = GetIntersectPointList(intersections)
        intersectPointList.append(shapely.Point(vertex))

        for intersectPoint in intersectPointList:
            intersectPoint = shapely.Point(
                MyRound(intersectPoint.x, tolerance), MyRound(intersectPoint.y, tolerance))
            image = DrawPoints(image, intersectPoint.x, intersectPoint.y)
            if (judgeVisible(polygon, watcher, intersectPoint)):
                image = DrawPoints(image, intersectPoint.x,
                                   intersectPoint.y, (0, 0, 255))
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


if __name__ == '__main__':
    polygon = random_polygons_generate.GetPolygon(30)
    # polygon = shapely.Polygon([(0, 0), (1, 0), (1, 0.3), (0.5, 0.3),
    #                            (0.5, 0.6), (1, 0.6), (1, 1), (0, 1)])
    watcher = None
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if polygon.contains(p):
            watcher = p
            break
    # watcher = shapely.Point(0.505, 0.9999)
    # watcher = shapely.Point(0.5, 0.1)
    visibilityPolygon = GetVisibilityPolygon(
        polygon, watcher)
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    DrawPolygon((pic_size, pic_size, 3), list(
                polygon.exterior.coords), (255, 255, 255), image)
    DrawPolygon((pic_size, pic_size, 3), list(
        visibilityPolygon.exterior.coords), (25, 55, 255), image)
    image = DrawPoints(image, watcher.x, watcher.y)

    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
