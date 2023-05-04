import cv2
import numpy as np
import shapely
from ..Global import *
def DrawPolygon( points, color, image):

    # list -> ndarray
    points = np.array(points)
    points *= pic_size
    points /= zoomRate
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        image = cv2.fillPoly(image, [points], color)
    else:
        image = cv2.fillPoly(image, points, color)

    return image


def DrawPoints(image, x, y, color=(153, 92, 0)):
    x = np.round(x*pic_size/zoomRate).astype(np.int32)
    y = np.round(y*pic_size/zoomRate).astype(np.int32)
    cv2.circle(image, (x, y), 1, color, 2)
    return image

def DrawSinglePoint(image, x, y, color=30):
    x = np.round(x*pic_size/zoomRate).astype(np.int32)
    y = np.round(y*pic_size/zoomRate).astype(np.int32)
    # cv2.circle(image, (x, y), 1, color, 1)
    image[y][x] = color
    return image

def DrawGridNum(image, x, y, num):

    x = np.round(x*pic_size/grid_size).astype(np.int32)
    y = np.round(y*pic_size/grid_size).astype(np.int32)
    s_num = str(num)
    cv2.putText(image, s_num, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                0.8, (0, 255, 0), 2)

def DrawLine(image, pt1, pt2,color = (0, 25, 255)):

    x1 = int(pt1[0] * pic_size / zoomRate)
    y1 = int(pt1[1] * pic_size / zoomRate)
    x2 = int(pt2[0] * pic_size / zoomRate)
    y2 = int(pt2[1] * pic_size / zoomRate)

    cv2.line(image, (x1, y1), (x2, y2), color, 1)

def DrawPath(image, path):
    i = 0
    while i < len(path)-1:
        DrawLine(image, path[i], path[i+1])
        i += 1

def DrawMultiline(image, multiLine,color = (0, 25, 255)):
    
    def drawSingleline(image,line,color = (0, 25, 255)):
        pointList = list(line.coords)
        length = len(pointList)
        for i in range(length-1):
            DrawLine(image,pointList[i],pointList[i+1],color)

    if(type(multiLine) == shapely.LineString):
        drawSingleline(image,multiLine,color)
    elif(type(multiLine) == shapely.Polygon):
        DrawPolygon(list(multiLine.exterior.coords),color)
    elif(type(multiLine) == shapely.MultiLineString):
        for line in list(multiLine.geoms):
            drawSingleline(image,line,color)
    elif(type(multiLine) == shapely.GeometryCollection):
        for geometry in multiLine.geoms:
            DrawMultiline(image,geometry)
    else:
        print(type(multiLine))
        print("unknown type")