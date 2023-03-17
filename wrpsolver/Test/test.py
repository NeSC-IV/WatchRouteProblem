#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
import getopt
import sys
import cv2

from . import random_polygons_generate
from . import vis_maps
from ..WRP_solver import WatchmanRouteProblemSolver
from .draw_pictures import *


def RunTest():
    edgeNum = 20
    iterationNum = 10
    coverageRate = 0.95

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
    polygon = random_polygons_generate.GetPolygon(edgeNum)
    polygon = shapely.Polygon(vis_maps.GetPolygon(1))

    polygonCoverList, sampleList,order, length, path = WatchmanRouteProblemSolver(
        polygon, coverageRate, iterationNum,30000)
    print("The number of convex polygonlen is " + str(len(polygonCoverList)))
    print(length)
    length = 0
    for sample in sampleList:
        length += len(sample)
    print(length)

    # 绘制生成的多边形
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    DrawPolygon( list(polygon.exterior.coords), (255, 255, 255), image)
    n = 0
    m = 255
    o = 255

    for p in polygonCoverList:
        p = p.simplify(0.05, preserve_topology=False)
        image = DrawPolygon( list(p.exterior.coords), (o, n, m), image)
        n += 55
        if (n >= 255):
            m -= 55
        if (m <= 0):
            o -= 55



    # 绘制sample 和 访问顺序
    for sample in sampleList:
        for point in sample:
            DrawPoints(image, point.x, point.y)

    for i in range(len(order)):
        DrawNum(image, order[i][0], order[i][1], i)

    for i in range(len(path)):
        DrawPath(image, path[i])
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

