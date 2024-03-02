import shapely
from wrpsolver.MACS.compute_visibility import GetVisibilityPolygonCPP
pointList = [(86.4, 48.2), (86.4, 49.489949493661165), (85.88994949366118, 50.0), (83.81005050633883, 50.0), (83.3, 49.489949493661165), (83.3, 24.438575568787762), (72.32799709613104, 24.438575568787762), (72.32799709613104, 54.43857556878776), (102.32799709613104, 54.43857556878776), (102.32799709613104, 48.2), (86.4, 48.2)]
watcher = shapely.Point(0,0)
visiblePolygon = shapely.Polygon(pointList)
visiblePolygon = GetVisibilityPolygonCPP(visiblePolygon, watcher)