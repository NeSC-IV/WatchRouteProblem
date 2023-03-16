import shapely
from shapely.ops import split
from random_polygons_generate import GetPolygon
from draw_pictures import *
# from polygons_coverage import FindVisibleRegion, SelectPointFromPolygon


def GetKernel(polygon, watcher):

    kernel = polygon
    reflexPointList = []
    pointList = list(polygon.exterior.coords)
    pointList.pop()  # 去除重复顶点

    # 模拟循环链表
    n = pointNum = len(pointList)
    while pointNum < 2 * n:
        point = pointList[(pointNum) % n]
        pointLeft = pointList[(pointNum - 1) % n]
        pointRight = pointList[(pointNum + 1) % n]
        rayLine = GetRayLine(point, pointRight)
        splitedPolygons = split(kernel, rayLine)
        isConvex = (((point[0] - pointLeft[0]) * (pointRight[1] - pointLeft[1]) -
                    (pointRight[0] - pointLeft[0]) * (point[1] - pointLeft[1])) > 0)
        if isConvex:
            reflexPointList.append(point)
        for p in list(splitedPolygons.geoms):
            if p.intersects(watcher):
                kernel = p
                break

        pointNum += 1
    return kernel, reflexPointList


def GetRayLine(watcher, vertex):
    xGap = vertex[0] - watcher[0]
    yGap = vertex[1] - watcher[1]
    # todo 注意多边形的边界
    if (xGap == 0):
        extendRate = MyRound(2*zoomRate/abs(yGap), tolerance)
        extendPoint1 = (watcher[0], watcher[1] + yGap*extendRate)
        extendPoint2 = (watcher[0], watcher[1] - yGap*extendRate)
    elif (yGap == 0):
        extendRate = MyRound(tolerance*zoomRate/abs(xGap), tolerance)
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


if __name__ == '__main__':
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)

    polygon = GetPolygon(22)
    watcher = SelectPointFromPolygon(polygon)
    visiblePolygon = FindVisibleRegion(
        polygon, watcher, 1000000000)  # d为可视距离

    DrawPolygon((pic_size, pic_size, 3), list(
        polygon.exterior.coords), (255, 255, 255), image)
    DrawPolygon((pic_size, pic_size, 3), list(
        visiblePolygon.exterior.coords), (255, 0, 255), image)
    image = DrawPoints(image, watcher.x, watcher.y)

    kernel, reflexList = GetKernel(visiblePolygon,  watcher)
    print(kernel)
    DrawPolygon((pic_size, pic_size, 3), list(
        kernel.exterior.coords), (0, 255, 255), image)
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
