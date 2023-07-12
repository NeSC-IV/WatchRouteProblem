import numpy as np
tolerance = 3
zoomRate = 1000
pic_size = 100
step = zoomRate/pic_size
threadNum = 48
def MyRound(num, tolerance):
    fmt = '.' + str(tolerance) + 'f'
    return float(format(num, fmt))

