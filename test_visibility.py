from wrpsolver.MACS.polygons_coverage import FindVisibleRegion,SelectMaxPolygon
from wrpsolver.Test.draw_pictures import DrawMultiline,DrawPolygon,DrawPoints
import cv2
import shapely
import numpy as np
pointList = [[82.8, 0.8], [82.8, 1.8], [65.8, 1.8], [65.8, 27.8], [63.2, 27.8], [63.2, 1.8], [19.8, 1.8], [19.8, 9.8], [17.2, 9.8], [17.2, 8.8], [17, 8.8], 
             [17, 7.2], [17.2, 7.2], [17.2, 1.8], [0.8, 1.8], [0.8, 26.2], [22.2, 26.2], [22.2, 15.331370849898475], [20.068629150101522, 13.2], 
             [23.331370849898477, 13.2], [24.8, 14.668629150101525], [24.8,28.8], [1.8000000000000003, 28.8], [1.8000000000000003, 51.2], [10.2, 51.2], 
             [10.2, 50.2], [63.2,0.2],[ 63.2, 46.8], [62.2 ,46.8], [62.2, 40.2], [63.2 ,40.2], [63.2 ,35.2], [65.8 ,35.2], [65.8 ,38.2], [94.2 ,38.2], [94.2 ,0.8], 
             [82.8, 0.8]]
polygon = shapely.Polygon(pointList)
point = shapely.Point(46, 30)
print(not polygon.contains(point))
image = np.zeros((100, 100, 1), dtype=np.uint8)
DrawMultiline(image,polygon,color = (255))
DrawPoints(image,point.x,point.y,(30),-1)
cv2.imwrite('test.png',image)
vp = FindVisibleRegion(polygon,point,32,True)