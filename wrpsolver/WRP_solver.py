import shapely
import cv2
import numpy as np
from func_timeout import func_set_timeout
from . import GTSP
from . import MACS
from .Global import step,pic_size
import time
import logging
# logging.basicConfig(level=logging.DEBUG)
@func_set_timeout(60)
def WatchmanRouteProblemSolver(polygon,coverage,iteration = 32,d = pic_size):
    gridMap = np.zeros((pic_size, pic_size, 1), dtype=np.uint8)
    polygon = polygon.simplify(0.0001, preserve_topology=False)
    time1 = time.time()
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    logging.debug(time.time() - time1)
    time1 = time.time()
    freeSpace = polygon.buffer(-(step*1))
    freeSpace = MACS.SelectMaxPolygon(freeSpace)
    gridMap = Polygon2Gird(freeSpace,255,gridMap)
    sampleList= GTSP.GetSample(convexSet,freeSpace,pic_size/10)
    gtspCase = GTSP.postProcessing(sampleList)
    logging.debug(time.time() - time1)
    time1 = time.time()
    order, length, path = GTSP.GetTrace(gtspCase,gridMap)
    logging.debug(time.time() - time1)
    return convexSet,sampleList,order,length,path
    
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