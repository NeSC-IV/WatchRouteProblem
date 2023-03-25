import shapely
from . import GTSP
from . import MACS
from .Global import *
import time
def WatchmanRouteProblemSolver(polygon,coverage,iteration = 10,d = zoomRate):
    
    polygon = shapely.Polygon(polygon)
    polygon = polygon.simplify(0.05, preserve_topology=False)
    time1 = time.time()
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    print(time.time() - time1)
    time1 = time.time()
    
    sampleList= GTSP.GetSample(convexSet,polygon,zoomRate/5)

    gtspCase = GTSP.postProcessing(sampleList)
    print(time.time() - time1)
    time1 = time.time()
    order, length, path = GTSP.GetTrace(gtspCase,polygon)
    print(time.time() - time1)
    time1 = time.time()
    return convexSet,sampleList,order,length,path
    