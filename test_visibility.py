import shapely
import visibility
import numpy as np
import cv2
from wrpsolver.MACS.compute_visibility import GetVisibilityPolygon
from wrpsolver.Test.draw_pictures import DrawPolygon,DrawPoints
from wrpsolver.MACS.polygons_coverage import SelectMaxPolygon
if __name__ =='__main__':
    polygon = [[40, 1], [40, 1], [40, 61], [84, 61], [84, 60], [84, 60], [85, 60], [85, 60], [86, 60], [86, 60], [86, 61], [86, 61], [86, 61], [86, 70], [86, 70], [84, 70], [84, 70], [84, 62], [38, 62], [38, 63], [1, 63], [1, 63], [1, 149], [55, 149], [55, 120], [55, 120], [55, 91], [55, 91], [55, 90], [56, 89], [57, 89], [57, 89], [84, 89], [84, 81], [84, 81], [86, 81], [86, 81], [86, 91], [86, 91], [86, 91], [86, 91], [85, 91], [85, 109], [85, 109], [85, 127], [85, 127], [85, 134], [87, 134], [87, 134], [87, 136], [87, 136], [85, 136], [85, 164], [85, 164], [85, 196], [165, 196], [165, 136], [98, 136], [98, 136], [98, 134], [98, 134], [165, 134], [165, 134], [165, 129], [166, 129], [166, 122], [166, 122], [166, 115], [166, 115], [166, 108], [166, 108], [166, 101], [166, 101], [166, 94], [166, 94], [166, 87], [166, 87], [166, 80], [167, 80], [167, 73], [167, 73], [167, 66], [167, 66], [167, 59], [167, 59], [167, 57], [125, 57], [125, 57], [125, 55], [125, 55], [169, 55], [169, 55], [199, 55], [199, 1], [115, 1], [115, 48], [115, 48], [114, 48], [114, 48], [114, 1], [86, 1], [86, 14], [86, 14], [86, 37], [86, 37], [86, 49], [86, 49], [84, 49], [84, 49], [84, 41], [84, 41], [84, 1]]
    startPoint = (55, 60)
    polygon = [(41.0, 1.0), (41.0, 2.0), (43.0, 2.0), (43.0, 1.0), (81.0, 1.0), (81.0, 64.0), (58.0, 64.0), (58.0, 66.0), (81.0, 66.0), (81.0, 74.0), (83.0, 74.0), (83.0, 37.0), (91.0, 37.0), (91.0, 35.0), (83.0, 35.0), (83.0, 1.0), (138.0, 1.0), (138.0, 35.0), (135.0, 35.0), (135.0, 37.0), (138.0, 37.0), (138.0, 54.0), (140.0, 54.0), (140.0, 29.0), (194.0, 29.0), (194.0, 96.0), (140.0, 96.0), (140.0, 71.0), (138.0, 71.0), (138.0, 96.0), (83.0, 96.0), (83.0, 91.0), (81.0, 91.0), (81.0, 96.0), (43.0, 96.0), (43.0, 87.0), (41.0, 87.0), (41.0, 96.0), (1.0, 96.0), (1.0, 39.0), (20.0, 39.0), (20.0, 37.0), (1.0, 37.0), (1.0, 1.0)]
    startPoint = (137.00001, 37)
    polygon = [(173.0, 2.0), (173.0, 152.0), (114.0, 152.0), (114.0, 40.0), (111.0, 40.0), (111.0, 152.0), (59.0, 152.0), (59.0, 111.0), (58.0, 111.0), (58.0, 64.0), (57.0, 63.0), (57.0, 40.0), (54.0, 40.0), (54.0, 63.0), (55.0, 64.0), (55.0, 111.0), (56.0, 111.0), (56.0, 152.0), (9.0, 152.0), (9.0, 153.0), (2.0, 153.0), (2.0, 3.0), (3.0, 2.0)]
    startPoint = (57.0, 31.0)
    polygon = shapely.Polygon(polygon)
    watcher = shapely.Point(startPoint)
    print(polygon.contains(watcher))
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image = DrawPolygon( list(polygon.exterior.coords), (100,255 , 255), image)
    DrawPoints(image,watcher.x,watcher.y)
    cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/visibility.png',image)
    # visibility = GetVisibilityPolygon(polygon,watcher)

    polygon = polygon.simplify(0.05,preserve_topology=False)
    pointList = list(polygon.exterior.coords)
    pointList.pop()
    try:
        result = visibility.compute_visibility_cpp(pointList,startPoint)
    finally:
        pass
    visibility = shapely.Polygon(result)
    # visibility = SelectMaxPolygon(visibility)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image = DrawPolygon( list(visibility.exterior.coords), (100,255 , 255), image)
    DrawPoints(image,watcher.x,watcher.y)
    cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/visibility.png',image)
