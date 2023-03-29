import shapely
tolerance = 2
zoomRate = 1000
step = zoomRate/200
def MyRound(num, tolerance):
    fmt = '.' + str(tolerance) + 'f'
    return float(format(num, fmt))


def GetIntPolygon(polygon):
    pointList = list(polygon.exterior.coords)
    pointList = [(MyRound(point[0], 0), MyRound(point[1], 0))
                 for point in pointList]
    return shapely.Polygon(pointList)
