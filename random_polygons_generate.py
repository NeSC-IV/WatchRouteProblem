#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
from draw_pictures import *
import cv2
import matplotlib.pyplot as plt
from shapely import Polygon
import random
# random.seed(11)

# function to save image
zoomRate = 10000


def PlotPolygon(polygon, out_file_name):
    plt.figure()
    plt.gca().set_aspect("equal")

    for i, (x, y) in enumerate(polygon):
        plt.text(x, y, str(i), horizontalalignment="center",
                 verticalalignment="center")

    # just so that it is plotted as closed polygon
    polygon.append(polygon[0])

    xs, ys = zip(*polygon)
    plt.plot(xs, ys, "r-", linewidth=0.4)

    plt.savefig(out_file_name, dpi=300)
    plt.close()

# function to show image


def GetPolygon(edgeNum: int):

    if edgeNum < 3:
        print("The number of edge must lager than 3 !")
        return None
    polygon = random_polygon(num_points=edgeNum)
    polygon = ((point[0]*zoomRate, point[1]*zoomRate) for point in polygon)
    polygon = ((format(point[0], '.1f'), format(
        point[1], '.1f')) for point in polygon)
    polygon = Polygon(polygon)
    return polygon


def GetInsidePolygon(polygon):
    center = polygon.representative_point()
    x = center.x
    y = center.y
    pointsList = ([-0.04+x, 0.04+y], [-0.04+x, -0.04+y],
                  [0.04+x, -0.04+y], [0.04+x, 0.04+y])
    inPolygon = Polygon(pointsList)
    return inPolygon


if __name__ == '__main__':
    # this is just so that you can reproduce the same results
    # random.seed(5)

    # generate points list for polygon
    polygon = GetPolygon(20)
    inPolygon = GetInsidePolygon(polygon)
    # draw pics
    image = np.zeros((pic_size, pic_size, 3), dtype=np.uint8)
    image = DrawPolygon((pic_size, pic_size, 3), list(
        polygon.exterior.coords), (255, 255, 255), image)
    image = DrawPolygon((pic_size, pic_size, 3), list(
        inPolygon.exterior.coords), (102, 0, 255), image)

    # newpolygon = polygon.difference(inPolygon)
    # print(inPolygon)
    # print(newpolygon)
    # image1 = np.zeros((pic_size,pic_size,3), dtype=np.uint8)
    # image1 = DrawPolygon((pic_size,pic_size,3), list(newpolygon.exterior.coords), (255,255,255),image1)

    cv2.imshow('a', image)
    cv2.waitKey(0)
    # cv2.imshow('b', image1)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
