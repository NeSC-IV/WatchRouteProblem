import shapely
import cv2
import numpy as np
from func_timeout import func_set_timeout
from . import GTSP
from . import MACS
from .Test.draw_pictures import DrawPolygon,DrawMultiline
import time
import logging
import math
logging.basicConfig(level=logging.INFO)
# @func_set_timeout(30)
@profile
def WatchmanRouteProblemSolver(polygon,coverage,d,iteration = 32,step = 3):
    d = d/1.7
    convexSet = []
    sampleList = []
    order = []
    length = 0
    path = []
    isSuccess = True

    if(type(polygon.buffer(-2, join_style=2)) != shapely.Polygon):
        isSuccess = False
        return convexSet,sampleList,order,length,path,isSuccess
    minx, miny, maxx, maxy = polygon.bounds
    maxx = math.ceil(maxx/10)*10
    maxy = math.ceil(maxy/10)*10
    gridMap = np.zeros((int(maxy), int(maxx)), dtype=np.uint8)
    gridMap = Polygon2Gird(polygon.buffer(-2, join_style=2),255,gridMap)

    time1 = time.time()
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    logging.debug(time.time() - time1)
    time1 = time.time()

    sampleList= GTSP.GetSample(convexSet, polygon, 20, gridMap, step=step)
    if (not len(convexSet)==len(sampleList))  or (len(convexSet) > 100) or (len(convexSet) <=3):
        isSuccess = False
        return convexSet,sampleList,order,length,path,isSuccess
    
    gtspCase = GTSP.postProcessing(sampleList)
    logging.debug(time.time() - time1)
    time1 = time.time()
    order, length, path = GTSP.GetTraceGLNS(gtspCase,gridMap,step)
    # order2, length2, path2 = GTSP.GetTraceACO(gtspCase,gridMap,step) 
    # order3, length3, path3 = GTSP.GetTraceTabu(gtspCase,gridMap,step) 

    logging.debug(time.time() - time1)
    return convexSet,sampleList,order,length,path,isSuccess
    
def Polygon2Gird(polygon, color, gridMap):

    points = list(polygon.exterior.coords)
    # list -> ndarray
    points = np.array(points)
    points = np.round(points).astype(np.int32)
    gridMap = cv2.fillPoly(gridMap, [points], color)

    holes = [list(interior.coords) for interior in polygon.interiors]
    for i in range(len(holes)):
        hole = holes[i]
        hole = np.array(hole)
        hole = np.round(hole).astype(np.int32)
        gridMap = cv2.fillPoly(gridMap, [hole], 0, 8)

    return gridMap