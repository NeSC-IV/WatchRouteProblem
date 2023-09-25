#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import getopt
import sys
import cv2
import shapely
import math
import shutil
import os
from . import random_polygons_generate
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from .draw_pictures import *


def RunTest(seed = 1):
    iterationNum = 32
    d = 20
    coverageRate = 0.97

    # 读取命令行参数
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'he:i:c:')
    except getopt.GetoptError:
        print("Usage:")
        print(
            "python/python3 polygons_coverage.py -e <num_of_edge> -i <num_of_iteration> {-c <coverage_rate>}")
        print("For example:")
        print("python3 polygons_coverage.py -e 20 -i 10 {-c 0.98}")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("Usage:")
            print(
                "python/python3 polygons_coverage.py -e <num_of_edge> -i <num_of_iteration>")
            print("For example:")
            print("python3 polygons_coverage.py -e 20 -i 10 {-c 0.98}")
            sys.exit(0)
        if opt == '-e':
            edgeNum = int(arg)
        elif opt == '-i':
            iterationNum = int(arg)
        elif opt == '-c':
            coverageRate = float(arg)

    # 随机生成多边形
    shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/test/")
    os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/test/")
    pointList,_,_ = vis_maps.GetPolygon(seed)
    polygon = shapely.Polygon(pointList).simplify(0.5,True).buffer(-0.7,join_style=2)
    minx, miny, maxx, maxy = polygon.bounds
    maxx = math.ceil(maxx/10)*10
    maxy = math.ceil(maxy/10)*10
    image = np.zeros((int(maxy), int(maxx), 3), dtype=np.uint8)
    DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image, zoomRate=1)
    cv2.imwrite('test/test.png',image)
    image = cv2.resize(image,(100,100),interpolation = cv2.INTER_NEAREST)
    cv2.imwrite('test/test0.png',image)

    polygonCoverList, sampleList,order, length, path, _ = WatchmanRouteProblemSolver(
        polygon, coverageRate, d, iterationNum)
    print("The number of convex polygonlen is " + str(len(polygonCoverList)))
    print(length)
    length = 0
    for sample in sampleList:
        length += len(sample)
    print(length)

    # 绘制生成的多边形
    DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image, zoomRate=1)
    cv2.imwrite('test/test1.png',image)
    colorList = []
    for n in range(192,-1,-64):
        for m in range(256,-1,-64):
            for o in range(256,-1,-64):
                colorList.append((o,m,n))
    cnt = 0
    for p in polygonCoverList:
        image = image.copy()
        p = p.simplify(0.05, preserve_topology=False)
        DrawPolygon( list(p.exterior.coords), colorList[cnt], image, zoomRate=1)
        cv2.imwrite('test/test2_'+ str(cnt) +'.png',image)
        cnt += 1

    # 绘制sample 和 访问顺序
    for sample in sampleList:
        for point in sample:
            DrawPoints(image, point.x, point.y,zoomRate=(1))
            pass
    cv2.imwrite('test/test3.png',image)

    for i in range(len(path)):
        pass
        DrawPath(image, path[i])
    cv2.imwrite('test/test4.png',image)

    for i in range(len(order)):
        # DrawGridNum(image, order[i][0], order[i][1], i)
        DrawPoints(image, order[i][0], order[i][1],(0,255,0),1)
    cv2.imwrite('test/test5.png',image)

