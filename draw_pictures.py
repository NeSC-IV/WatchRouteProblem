import cv2
import numpy as np
from _global import *


def DrawPolygon(image_size, points, color, image):
    """
    draw polygon(s) on a image

    Parameters:
    -----------
    image_size: a list/tuple of numbers
        image size = [image_height, image_width, image_channel]
    points: 2D ndarray or a list of 2D ndarray
        points that can construct a random polygon, also can be a list of
        points that can construct random polygons
    color: a list/tuple of numbers, whose length is same as image channel
        color of polygon

    Returns:
    --------
    image: ndarray
        image with polygon(s) on it
    """
    # list -> ndarray
    points = np.array(points)
    points /= zoomRate
    # point's position from (0-1) to (0-image_Size)
    points *= pic_size
    # float -> int32
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
                0.8, (255, 25, 255), 2)


def DrawLine(image, pt1, pt2):

    x1 = int(pt1[0] * pic_size / zoomRate)
    y1 = int(pt1[1] * pic_size / zoomRate)
    x2 = int(pt2[0] * pic_size / zoomRate)
    y2 = int(pt2[1] * pic_size / zoomRate)

    cv2.line(image, (x1, y1), (x2, y2), (122, 234, 31), 2, 8)
