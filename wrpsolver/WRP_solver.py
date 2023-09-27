import shapely
import cv2
import numpy as np
from func_timeout import func_set_timeout
from . import GTSP
from . import MACS
from .Global import step,pic_size
from .Test.draw_pictures import DrawPolygon,DrawMultiline
import time
import logging
import math
logging.basicConfig(level=logging.DEBUG)
@func_set_timeout(6000)
def WatchmanRouteProblemSolver(polygon,coverage,d,iteration = 32):
    convexSet = []
    sampleList = []
    order = []
    length = 0
    path = []
    isSuccess = True
    time1 = time.time()
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    logging.debug(time.time() - time1)
    time1 = time.time()

    sampleList= GTSP.GetSample(convexSet, polygon, 15)
    if not (len(convexSet)==len(sampleList)):
        isSuccess = False
        return convexSet,sampleList,order,length,path,isSuccess
    minx, miny, maxx, maxy = polygon.bounds
    maxx = math.ceil(maxx/10)*10
    maxy = math.ceil(maxy/10)*10
    gridMap = np.zeros((int(maxy), int(maxx)), dtype=np.uint8)
    gridMap = Polygon2Gird(polygon.buffer(-1, join_style=2),255,gridMap)

    
    gtspCase = GTSP.postProcessing(sampleList)
    logging.debug(time.time() - time1)
    time1 = time.time()
    order, length, path = GTSP.GetTrace(gtspCase,gridMap)
    logging.debug(time.time() - time1)
    return convexSet,sampleList,order,length,path,isSuccess
    
def Polygon2Gird(polygon, color, gridMap):

    points = list(polygon.exterior.coords)
    # list -> ndarray
    points = np.array(points)
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        gridMap = cv2.fillPoly(gridMap, [points], color)
    else:
        gridMap = cv2.fillPoly(gridMap, points, color)

    return gridMap