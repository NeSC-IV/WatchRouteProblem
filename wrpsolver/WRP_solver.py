import shapely
from . import GTSP
from . import MACS
from .Global import *
def WatchmanRouteProblemSolver(polygon,coverage,iteration = 10,d = zoomRate):
    polygon = shapely.Polygon(polygon)
    convexSet = MACS.PolygonCover(polygon,d,coverage,iteration)
    sampleList= GTSP.GetSample(convexSet,polygon,zoomRate/10)
    gtspCase = GTSP.postProcessing(sampleList)
    order, length, path = GTSP.GetTrace(gtspCase,polygon)
    return convexSet,sampleList,order,length,path
    