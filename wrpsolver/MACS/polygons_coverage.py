#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import shapely
import random
from shapely.ops import split, nearest_points
from shapely.validation import make_valid
from multiprocessing import Pool

from .compute_kernel import GetKernel
from .compute_visibility import GetVisibilityPolygon,GetVisibilityPolygonCPP
from ..Global import *


def SelectMaxPolygon(polygons):
    MaxPolygon = shapely.Point(1, 1)
    if (type(polygons) == shapely.Polygon):
        MaxPolygon = polygons
    else:
        for p in list(polygons.geoms):
            if (type(p) == shapely.MultiPolygon):
                p = SelectMaxPolygon(p)
            if p.area > MaxPolygon.area and (type(p) == shapely.Polygon):
                MaxPolygon = p
    return MaxPolygon


def SelectPointFromPolygon(polygon):
    if not shapely.is_geometry(polygon):
        print("Polygon must be created by lib shapely !")
        return None
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = shapely.Point(random.uniform(minx, maxx),
                          random.uniform(miny, maxy))
        if polygon.contains(p):
            return p

def FindVisibleRegion(polygon, watcher, d, useCPP = False):

    try:
        if(useCPP):
            visiblePolygon = GetVisibilityPolygonCPP(polygon, watcher)
        else:
            visiblePolygon = GetVisibilityPolygon(polygon, watcher)
        return SelectMaxPolygon(visiblePolygon)
        # visiblePolygon = make_valid(visiblePolygon)
        # dVisibility = watcher.buffer(d)  # d范围视距
        # finalVisibility = visiblePolygon.intersection(dVisibility)  # 有限视距下的可视范围

        # return SelectMaxPolygon(finalVisibility)
    except Exception as e:
        print(e)
        return None



def GetKernelPolygon(visiblePolygon):
    xList, yList = GetKernel(visiblePolygon)
    kernel = list(zip(xList, yList))
    return shapely.Polygon(kernel)


def GetRayLine(watcher, vertex):
    xGap = vertex[0] - watcher[0]
    yGap = vertex[1] - watcher[1]
    if (xGap == 0):
        extendRate = MyRound(2*zoomRate/abs(yGap), tolerance)
        extendPoint1 = (watcher[0], watcher[1] + yGap*extendRate)
        extendPoint2 = (watcher[0], watcher[1] - yGap*extendRate)
    elif (yGap == 0):
        extendRate = MyRound(2*zoomRate/abs(xGap), tolerance)
        extendPoint1 = (watcher[0] + xGap*extendRate, watcher[1])
        extendPoint2 = (watcher[0] - xGap*extendRate, watcher[1])
    else:
        extendRate = max(zoomRate/abs(xGap), zoomRate/abs(yGap))
        extendRate = MyRound(extendRate, tolerance)
        extendPoint1 = (
            MyRound(watcher[0] + xGap*extendRate, tolerance), MyRound(watcher[1] + yGap*extendRate, tolerance))
        extendPoint2 = (
            MyRound(watcher[0] - xGap*extendRate, tolerance), MyRound(watcher[1] - yGap*extendRate, tolerance))
    return shapely.LineString([extendPoint1, extendPoint2])


def GetSingleReflexChord(visiblePolygonPointList, reflexPoint, kernel):

    prependicular = None
    kernelPointList = list(kernel.exterior.coords)
    kernelPointList.pop()
    # center = tuple(map(operator.truediv, reduce(
    # lambda x, y: map(operator.add, x, y), kernelPointList), [len(kernelPointList)] * 2))
    # kernelPointList = (sorted(kernelPointList, key=lambda kernelPointList: (-135 - math.degrees(math.atan2(*
    #    tuple(map(operator.sub, kernelPointList, center))[::-1]))) % 360, reverse=True))
    kernelPointList.reverse()
    if (reflexPoint in list(kernelPointList)):  # 如果反射点在kernel上
        numOfKernelPoints = len(kernelPointList)
        reflex_kernel_pos = visiblePolygonPointList.index(reflexPoint)

        reflexKernelLeft = kernelPointList[(
            reflex_kernel_pos - 1) % numOfKernelPoints]
        reflexKernelRight = kernelPointList[(
            reflex_kernel_pos + 1) % numOfKernelPoints]
        # 和弦的斜率应为 反射点临边斜率的平均值
        if (abs(reflexPoint[0] - reflexKernelLeft[0]) < 1e-2):
            if (abs(reflexKernelRight[0] - reflexPoint[0]) < 1e-2):
                prependicular = 1e+2
            else:
                prependicular = 2 * \
                    (reflexKernelRight[1] - reflexPoint[1]) / \
                    (reflexKernelRight[0] - reflexPoint[0])
        else:
            if (abs(reflexKernelRight[0] - reflexPoint[0]) < 1e-2):
                prependicular = 2 * \
                    (reflexPoint[1] - reflexKernelLeft[1]) / \
                    (reflexPoint[0] - reflexKernelLeft[0])
            else:
                prependicular1 = (
                    reflexKernelRight[1] - reflexPoint[1]) / (reflexKernelRight[0] - reflexPoint[0])
                prependicular2 = (
                    reflexPoint[1] - reflexKernelLeft[1]) / (reflexPoint[0] - reflexKernelLeft[0])

                prependicular = (prependicular1 + prependicular2) / 2

    else:
        point = shapely.Point(reflexPoint)  # 如果反射点不在kernel上
        nearestPoint = (nearest_points(point, kernel))[1]
        # 和弦的斜率应为 点连线的垂线的斜率
        if abs(point.y - nearestPoint.y) < 1e-6:  # 斜率判断
            prependicular = 1e+6

        else:
            prependicular = (nearestPoint.x - point.x) / \
                (point.y - nearestPoint.y)
    extendPoint = (reflexPoint[0]+1, reflexPoint[1] + prependicular)
    return GetRayLine(reflexPoint, extendPoint)


def GetSplitedPolygon(chord, visiblePolygon, watcher):
    tempVisiblePolygon = visiblePolygon
    polygons = list(split(tempVisiblePolygon, chord).geoms)
    for polygon in polygons:
        if polygon.buffer(zoomRate/100).covers(watcher):
            return polygon
    print(polygons)


def MaximallyCoveringConvexSubset(args):  # MCCS
    unCoveredPolygon = args[0]
    initialPolygon = args[1]
    watcher = args[2]
    d = args[3]/2

    visiblePolygon = FindVisibleRegion(
        initialPolygon, watcher, d)  # d为可视距离
    if not (visiblePolygon.buffer(zoomRate/100).covers(watcher)):
        print("error")
        return shapely.Point(1,1)
    kernelPolygon, reflexPointList = GetKernel(visiblePolygon, watcher)
    reflexPointList.sort(key=lambda point: shapely.distance(
        kernelPolygon, shapely.Point(point)))  # 列表排序

    polygon = visiblePolygon
    numOfReflexPoints = len(reflexPointList)

    for i in range(numOfReflexPoints):

        polygonPointList = list(polygon.exterior.coords)
        polygonPointList.pop()
        numOfPolygonPoints = len(polygonPointList)
        reflexPoint1 = reflexPointList[i]
        reflexPoint2 = reflexPointList[(i+1) % numOfReflexPoints]

        if (reflexPoint1 not in polygonPointList):
            continue
        r1Pos = polygonPointList.index(reflexPoint1)
        r1Left = polygonPointList[(r1Pos - 1) % numOfPolygonPoints]
        r1Right = polygonPointList[(r1Pos + 1) % numOfPolygonPoints]

        # extremal chords
        chord = GetRayLine(reflexPoint1, r1Left)
        ePolygon1 = GetSplitedPolygon(chord, polygon, watcher)

        chord = GetRayLine(reflexPoint1, r1Right)
        ePolygon2 = GetSplitedPolygon(chord, polygon, watcher)

        # two reflex chord
        if (len(reflexPointList) > 1):
            chord = GetRayLine(reflexPoint1, reflexPoint2)
            tPolygon = GetSplitedPolygon(chord, polygon, watcher)
        else:
            tPolygon = shapely.Point(1, 1)  # area of point is 0

        # single reflex chord
        chord = GetSingleReflexChord(
            polygonPointList, reflexPoint1, kernelPolygon)
        sPolygon = GetSplitedPolygon(chord, polygon, watcher)
        polygon = max(ePolygon1, ePolygon2, tPolygon, sPolygon, key=lambda inptPolygon: (
            unCoveredPolygon.intersection(inptPolygon)).area)

    return polygon


def PolygonCover(polygon, d, coverage, iterations=16):
    polygonCoverList = []
    unCoverPolygon = shapely.Polygon(polygon)
    while ((unCoverPolygon.area / polygon.area) > (1-coverage)):

        point = SelectPointFromPolygon(unCoverPolygon)
        R0 = MaximallyCoveringConvexSubset((unCoverPolygon, polygon, point, d))
        bestR = R0
        #迭代开始
        num = iterations
        pointList = []
        pool = Pool(threadNum)
        while num > 0:
            pointList.append(SelectPointFromPolygon(R0))
            num -= 1
        RList =(pool.map(MaximallyCoveringConvexSubset,[(unCoverPolygon, polygon, point, d) for point in pointList]))
        RList.append(R0)
        bestR = max(RList,key=lambda R:(R.intersection(unCoverPolygon)).area)

        polygonCoverList.append(bestR)
        unCoverPolygon = unCoverPolygon.difference(bestR)

    return polygonCoverList
