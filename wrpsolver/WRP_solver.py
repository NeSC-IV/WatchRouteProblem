import shapely
import cv2
import numpy as np
from func_timeout import func_set_timeout
from . import GTSP
from . import MACS
from .Global import step,grid_size,zoomRate
import time
@func_set_timeout(30)
def WatchmanRouteProblemSolver(polygon,coverage,iteration = 10,d = zoomRate):
    grid = np.zeros((grid_size, grid_size, 1), dtype=np.uint8)
    polygon = shapely.Polygon(polygon)
    polygon = polygon.simplify(0.05, preserve_topology=False)
    time1 = time.time()
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    print(time.time() - time1)
    time1 = time.time()
    freeSpace = polygon.buffer(-(step*2))
    freeSpace = MACS.SelectMaxPolygon(freeSpace)
    grid = Polygon2Gird(freeSpace,255,grid)
    sampleList= GTSP.GetSample(convexSet,freeSpace,step*20)
    gtspCase = GTSP.postProcessing(sampleList)
    print(time.time() - time1)
    time1 = time.time()
    order, length, path = GTSP.GetTrace(gtspCase,grid)
    print(time.time() - time1)
    time1 = time.time()
    return convexSet,sampleList,order,length,path
    
def Polygon2Gird(polygon, color, grid):

    points = list(polygon.exterior.coords)
    # list -> ndarray
    points = np.array(points)
    points *= grid_size
    points /= zoomRate
    points = np.round(points).astype(np.int32)

    if type(points) is np.ndarray and points.ndim == 2:
        grid = cv2.fillPoly(grid, [points], color)
    else:
        grid = cv2.fillPoly(grid, points, color)

    return grid