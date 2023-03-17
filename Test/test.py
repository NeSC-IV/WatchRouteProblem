import getopt
import sys
import cv2

import random_polygons_generate
from ..WRP_solver import WatchmanRouteProblemSolver
from draw_pictures import *
def main():
    edgeNum = 20
    iterationNum = 10
    coverageRate = 0.98

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
    # polygon = shapely.Polygon([(0, 0), (10000, 0), (10000, 3000), (8000, 3000),
    #                            (8000, 6000), (10000, 6000), (10000, 10000), (0, 10000), (0, 6000), (2000, 6000), (2000, 3000), (0, 3000)])



    polygonCoverList,order, length, path = WatchmanRouteProblemSolver(polygon, 3000, 1-coverageRate, iterationNum)
    print("The number of convex polygonlen is " + str(len(polygonCoverList)))



        # 绘制生成的多边形
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    DrawPolygon((pic_size, pic_size, 3), list(
        polygon.exterior.coords), (255, 255, 255), image)
    n = 0
    m = 255
    o = 255

    for p in polygonCoverList:
        p = p.simplify(0.05, preserve_topology=False)
        image = DrawPolygon((pic_size, pic_size, 3), list(
            p.exterior.coords), (o, n, m), image)
        n += 55
        if (n >= 255):
            m -= 55
        if (m <= 0):
            o -= 55

    # 在凸子集上采样
    # length = 0
    # sampleList = GetSample(
    #     polygonCoverList, polygon, 3000, image)
    # for sample in sampleList:
    #     length += len(sample)
    # print(length)


    # 确定样本访问顺序
    # 绘制sample 和 访问顺序
    # for sample in sampleList:
    #     for point in sample:
    #         DrawPoints(image, point.x, point.y)

    for i in range(len(order)):
        DrawNum(image, order[i][0], order[i][1], i)

    for i in range(len(path)):
        DrawPath(image, path[i])
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()