import visilibity as vis
import random_polygons_generate
import matplotlib.pylab as p
import shapely
from shapely.ops import split
from shapely.ops import nearest_points

from functools import reduce
import operator
import math
# from polygons_coverage import SelectPointFromPolygon
from draw_pictures import *
from convex_subset import GetKernel
from convex_subset import GetReflexPointList


def save_print(polygon):
    end_pos_x = []
    end_pos_y = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()

        end_pos_x.append(x)
        end_pos_y.append(y)

    return end_pos_x, end_pos_y


def ShapelyPolygon_2_VisilibityPolygon(polygon):
    point_list = []
    pointsOfPolygon = list(polygon.exterior.coords)
    pointsOfPolygon.pop()
    pointsOfPolygon.reverse()
    # center = tuple(map(operator.truediv, reduce(lambda x, y: map(
    #     operator.add, x, y), pointsOfPolygon), [len(pointsOfPolygon)] * 2))
    # pointsOfPolygon = (sorted(pointsOfPolygon, key=lambda pointsOfPolygon: (-135 - math.degrees(
    #     math.atan2(*tuple(map(operator.sub, pointsOfPolygon, center))[::-1]))) % 360, reverse=True))
    print(pointsOfPolygon)
    for point in pointsOfPolygon:
        x = point[0]
        y = point[1]
        vis_point = vis.Point(x, y)
        point_list.append(vis_point)
    visilibityPolygon = vis.Polygon(point_list)
    return visilibityPolygon


def VisilibityPolygon_2_ShapelyPolygon(polygon):
    point_list = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()
        point_list.append((x, y))
    shapelyPolygon = shapely.Polygon(point_list)
    return shapelyPolygon


def test_kernel():
    polygon = random_polygons_generate.GetPolygon(10)
    # inPolygon = random_polygons_generate.GetInsidePolygon(polygon)
    watcher = SelectPointFromPolygon(polygon)

    # 创建多边形
    epsilon = 0.0000001
    point_list = []
    walls_y = []
    walls_x = []
    d = 0.5  # 视线大小

    pointsOfPolygon = list(polygon.exterior.coords)
    for point in pointsOfPolygon:
        x = point[0]
        y = point[1]
        vis_point = vis.Point(x, y)
        point_list.append(vis_point)
        walls_x.append(x)
        walls_y.append(y)

    walls = vis.Polygon(point_list)

    # 观察者
    observer = vis.Point(watcher.x, watcher.y)

    # 可视图
    env = vis.Environment([walls])
    isovist = vis.Visibility_Polygon(observer, env, epsilon)
    visiblePolygon = VisilibityPolygon_2_ShapelyPolygon(isovist)
    # dVisibility =  watcher.buffer(d)
    # finalVisibility = visiblePolygon.intersection(dVisibility)
    finalVisibility = visiblePolygon
    x_list, y_list = GetKernel(finalVisibility)
    outpt = list(zip(x_list, y_list))
    print(outpt)
    # paint
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    image = DrawPolygon((pic_size, pic_size, 3),
                        pointsOfPolygon, (255, 255, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3), list(
        finalVisibility .exterior.coords), (102, 0, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3), outpt, (102, 0, 0), image)
    # image = DrawPolygon((pic_size,pic_size,3), list(visiblePolygon.exterior.coords), (102,0,255),image)
    image = DrawPoints(image, watcher.x, watcher.y)
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return list(finalVisibility .exterior.coords)


def test_visibility():
    polygon = random_polygons_generate.GetPolygon(20)
    watcher = SelectPointFromPolygon(polygon)

    # 创建多边形
    epsilon = 0.0000001
    point_list = []
    walls_y = []
    walls_x = []
    d = 10  # 视线大小

    pointsOfPolygon = list(polygon.exterior.coords)
    for point in pointsOfPolygon:
        x = point[0]
        y = point[1]
        vis_point = vis.Point(x, y)
        point_list.append(vis_point)
        walls_x.append(x)
        walls_y.append(y)

    walls = vis.Polygon(point_list)

    # 观察者
    observer = vis.Point(watcher.x, watcher.y)

    # 可视图
    env = vis.Environment([walls])
    isovist = vis.Visibility_Polygon(observer, env, epsilon)
    visiblePolygon = VisilibityPolygon_2_ShapelyPolygon(isovist)
    dVisibility = watcher.buffer(d)
    finalVisibility = visiblePolygon.intersection(dVisibility)
    print(finalVisibility.boundary)
    # 画图
    point_x, point_y = save_print(isovist)
    point_x.append(isovist[0].x())
    point_y.append(isovist[0].y())
    p.plot([observer.x()], [observer.y()], 'go')
    p.plot(point_x, point_y)
    p.plot(walls_x, walls_y, 'black')
    # p.show()

    #
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    image = DrawPolygon((pic_size, pic_size, 3),
                        pointsOfPolygon, (255, 255, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3), list(
        finalVisibility .exterior.coords), (102, 0, 255), image)
    # image = DrawPolygon((pic_size,pic_size,3), list(visiblePolygon.exterior.coords), (102,0,255),image)
    image = DrawPoints(image, watcher.x, watcher.y)
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_reflex():
    polygon = random_polygons_generate.GetPolygon(20)
    # inPolygon = random_polygons_generate.GetInsidePolygon(polygon)
    watcher = SelectPointFromPolygon(polygon)

    # 创建多边形
    epsilon = 0.0000001
    point_list = []
    walls_y = []
    walls_x = []
    d = 0.5  # 视线大小

    pointsOfPolygon = list(polygon.exterior.coords)
    for point in pointsOfPolygon:
        x = point[0]
        y = point[1]
        vis_point = vis.Point(x, y)
        point_list.append(vis_point)
        walls_x.append(x)
        walls_y.append(y)

    walls = vis.Polygon(point_list)

    # 观察者
    observer = vis.Point(watcher.x, watcher.y)

    # 可视图
    env = vis.Environment([walls])
    isovist = vis.Visibility_Polygon(observer, env, epsilon)
    visiblePolygon = VisilibityPolygon_2_ShapelyPolygon(isovist)
    finalVisibility = visiblePolygon
    # dVisibility =  watcher.buffer(d)
    # finalVisibility = visiblePolygon.intersection(dVisibility)
    x_list, y_list = GetKernel(finalVisibility)
    kernel = list(zip(x_list, y_list))

    reflex_list = GetReflexPointList(finalVisibility)
    visibility_point_list = list(finalVisibility.exterior.coords)

    def cmp(point):
        p = shapely.Point(point)
        return shapely.distance(shapely.Polygon(kernel), p)
    reflex_list.sort(key=cmp)

    cutted_polygon = test_cut(reflex_list, watcher,
                              finalVisibility, shapely.Polygon(kernel))

    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    image = DrawPolygon((pic_size, pic_size, 3),
                        pointsOfPolygon, (255, 255, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3),
                        visibility_point_list, (102, 0, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3), list(
        cutted_polygon.exterior.coords), (102, 255, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3), kernel, (102, 0, 0), image)

    image = DrawPoints(image, watcher.x, watcher.y)
    n = 0
    for point in reflex_list:
        DrawNum(image, point[0], point[1], n)
        n += 1
    cv2.imshow('polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return list(finalVisibility .exterior.coords)


def test_cut(reflex_list: list, watcher, visibility_polygon, kernel):
    polygon = visibility_polygon
    polygon_point_list = list(polygon.exterior.coords)
    polygon_point_list.pop()

    num_of_polygon_points = len(polygon_point_list)
    num_of_reflex_points = len(reflex_list)

    def get_ray_line(p1, p2):
        x_gap = p2[0] - p1[0]
        y_gap = p2[1] - p1[1]
        extend_rate = max(1.0/(abs(x_gap)+0.000001), 1.0/(abs(x_gap)+0.000001))

        # extend_rate = max(abs(1.0/x_gap),abs(1.0/y_gap))
        result1 = (p1[0] + x_gap*extend_rate, p1[1] + y_gap*extend_rate)

        x_gap = p1[0] - p2[0]
        y_gap = p1[1] - p2[1]
        extend_rate = max(1.0/(abs(x_gap)+0.000001), 1.0/(abs(x_gap)+0.000001))
        result2 = (p2[0] + x_gap*extend_rate, p2[1] + y_gap*extend_rate)
        return shapely.LineString([result1, result2])

    def get_single_reflex_chord(reflex_point, kernel):

        prependicular = None
        kernel_point_list = list(kernel.exterior.coords)
        kernel_point_list.pop()
        if (reflex_point in kernel_point_list):  # 如果反射点在kernel上
            num_of_kernel_points = len(kernel_point_list)
            reflex_kernel_pos = polygon_point_list.index(reflex_point)

            reflex_kernel_left = kernel_point_list[(
                reflex_kernel_pos - 1) % num_of_kernel_points]
            reflex_kernel_right = kernel_point_list[(
                reflex_kernel_pos + 1) % num_of_kernel_points]
            # todo 加入斜率为零时的判断
            prependicular = (2*reflex_point[1] - reflex_kernel_left[1] - reflex_kernel_right[1]) / (
                2*reflex_point[0] - reflex_kernel_left[0] - reflex_kernel_right[0])
        else:
            point = shapely.Point(reflex_point)
            nearest_point = (nearest_points(point, kernel))[1]
            prependicular = (nearest_point.x - point.x) / \
                (point.y - nearest_point.y)
        point2 = (reflex_point[0], reflex_point[1] + prependicular)
        return get_ray_line(reflex_point, point2)

    def get_splited_polygon(chord, visibility_polygon):
        polygons = list(split(visibility_polygon, chord).geoms)
        for polygon in polygons:
            if polygon.contains(watcher):
                return polygon

    for i in range(num_of_reflex_points):
        # global polygon,polygon_point_list
        r1 = reflex_list[i]
        r2 = reflex_list[(i+1) % num_of_reflex_points]

        if (r1 not in polygon_point_list):
            continue
        r1_pos = polygon_point_list.index(r1)
        r1_left = polygon_point_list[(r1_pos - 1) % num_of_polygon_points]
        r1_right = polygon_point_list[(r1_pos + 1) % num_of_polygon_points]

        # extremal chords
        chord = get_ray_line(r1, r1_left)
        e_polygon1 = get_splited_polygon(chord, polygon)

        chord = get_ray_line(r1, r1_right)
        e_polygon2 = get_splited_polygon(chord, polygon)

        # two reflex chord
        if (len(reflex_list) > 1):
            chord = get_ray_line(r1, r2)
            t_polygon = get_splited_polygon(chord, polygon)
        else:
            t_polygon = shapely.Point(1, 1)  # area of point is 0

        # single reflex chord
        chord = get_single_reflex_chord(r1, kernel)
        s_polygon = get_splited_polygon(chord, polygon)
        print(s_polygon)

        def cmp(inpt):
            return inpt.area
        polygon = max(e_polygon1, e_polygon2, t_polygon, s_polygon, key=cmp)
        polygon_point_list = list(polygon.exterior.coords)
        polygon_point_list.pop()
        num_of_polygon_points = len(polygon_point_list)

    return polygon


if __name__ == '__main__':
    test_reflex()
    # test_kernel()
    # test_visibility()
