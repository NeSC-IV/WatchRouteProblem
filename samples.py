import shapely
import getopt
import sys
import random_polygons_generate
from polygons_coverage import PolygonCover, SelectPointFromPolygon
from draw_pictures import *
from gtsp import TaPu


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

            # start = shapely.get_point(line, 0)
            # if freeSpace.buffer(-20).contains(start):
            #     pointList.append(start)
            # end = shapely.get_point(line, -1)
            # if freeSpace.buffer(-20).contains(end):
            #     pointList.append(end)
        # path = 0
        # while (path < lineString.length):
        #     point = shapely.line_interpolate_point(lineString, path)
        #     if freeSpace.buffer(-20).contains(point):
        #         pointList.append(point)
        #     path += dSample

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


if __name__ == '__main__':
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

    # 随机生成多边形和守卫点
    polygon = random_polygons_generate.GetPolygon(edgeNum)
    # polygon = shapely.Polygon([(0, 0), (10000, 0), (10000, 3000), (8000, 3000),
    #                            (8000, 6000), (10000, 6000), (10000, 10000), (0, 10000), (0, 6000), (2000, 6000), (2000, 3000), (0, 3000)])
    watcher = SelectPointFromPolygon(polygon)

    # 绘制生成的多边形
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    DrawPolygon((pic_size, pic_size, 3), list(
        polygon.exterior.coords), (255, 255, 255), image)
    image = DrawPoints(image, watcher.x, watcher.y)
    # cv2.imshow('polygons', image)
    # print("Press any key to continue!")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 求出并绘制最大凸子集
    polygonCoverList = PolygonCover(
        polygon, 3000, 1-coverageRate, iterationNum)
    print("The number of convex polygonlen is " + str(len(polygonCoverList)))
    n = 0
    m = 255
    o = 255
    # image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)

    freeSpace = shapely.Polygon()
    for p in polygonCoverList:
        p = p.simplify(0.05, preserve_topology=False)
        image = DrawPolygon((pic_size, pic_size, 3), list(
            p.exterior.coords), (o, n, m), image)
        n += 55
        if (n >= 255):
            m -= 55
        if (m <= 0):
            o -= 55
        freeSpace = freeSpace.union(p)

    # 在凸子集上采样
    length = 0
    sampleList = GetSample(
        polygonCoverList, polygon, 3000, image)
    for sample in sampleList:
        length += len(sample)
    print(length)


    # 确定样本访问顺序
    case = postProcessing(sampleList)
    order, length, path = TaPu(case, polygon)
    print(length)
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
