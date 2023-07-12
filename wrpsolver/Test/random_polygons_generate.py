#!/usr/bin/python3.8
# -*- coding:utf-8 -*-
from .draw_pictures import *
from shapely import Polygon
import random
# random.seed(110)
from ..Global import *



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
