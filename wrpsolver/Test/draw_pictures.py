import cv2
import numpy as np
from ..Global import *
pic_size = 1024
def DrawPolygon( points, color, image):

    # list -> ndarray
    points = np.array(points)
    points /= zoomRate
    points *= pic_size
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        image = cv2.fillPoly(image, [points], color)
    else:
        image = cv2.fillPoly(image, points, color)

    return image


def DrawPoints(image, x, y, color=(153, 92, 0)):
    x /= zoomRate
    y /= zoomRate
    x = np.round(x*pic_size).astype(np.int32)
    y = np.round(y*pic_size).astype(np.int32)
    cv2.circle(image, (x, y), 4, color, 8)
    return image


def DrawNum(image, x, y, num):
    x /= zoomRate
    y /= zoomRate
    x = np.round(x*pic_size).astype(np.int32)
    y = np.round(y*pic_size).astype(np.int32)
    s_num = str(num)
    cv2.putText(image, s_num, (x, y), cv2.FONT_HERSHEY_COMPLEX,
                0.8, (255, 255, 255), 2)


def DrawLine(image, pt1, pt2):

    x1 = int(pt1[0] * pic_size / zoomRate)
    y1 = int(pt1[1] * pic_size / zoomRate)
    x2 = int(pt2[0] * pic_size / zoomRate)
    y2 = int(pt2[1] * pic_size / zoomRate)

    cv2.line(image, (x1, y1), (x2, y2), (122, 234, 31), 2, 8)


def DrawPath(image, path):
    i = 0
    while i < len(path)-1:
        DrawLine(image, path[i], path[i+1])
        i += 1
