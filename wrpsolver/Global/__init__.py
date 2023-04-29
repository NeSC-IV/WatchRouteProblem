import numpy as np
tolerance = 2
zoomRate = 200
step = zoomRate/200
pic_size = 200
grid_size = 200
threadNum = 48
def MyRound(num, tolerance):
    fmt = '.' + str(tolerance) + 'f'
    return float(format(num, fmt))

