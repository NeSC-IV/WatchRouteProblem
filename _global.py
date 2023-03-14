tolerance = 2
pic_size = 1024

zoomRate = 10000


def MyRound(num, tolerance):
    fmt = '.' + str(tolerance) + 'f'
    return float(format(num, fmt))
