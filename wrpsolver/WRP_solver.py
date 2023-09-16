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
# logging.basicConfig(level=logging.DEBUG)
@func_set_timeout(600)
def WatchmanRouteProblemSolver(polygon,coverage,d,iteration = 32):
    convexSet = []
    sampleList = []
    order = []
    length = 0
    path = []
    isSuccess = True
    gridMap = np.zeros((pic_size, pic_size, 1), dtype=np.uint8)
    time1 = time.time()
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    logging.debug(time.time() - time1)
    time1 = time.time()
    freeSpace = polygon.buffer(-step,join_style=2)
    freeSpace = MACS.SelectMaxPolygon(freeSpace)
    gridMap = Polygon2Gird(freeSpace,255,gridMap)
    sampleList= GTSP.GetSample(convexSet,freeSpace,pic_size/5)
    if not (len(convexSet)==len(sampleList)):
        isSuccess = False
        return convexSet,sampleList,order,length,path,isSuccess
    
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