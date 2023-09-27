import numpy as np
from . import Astar

def RecordDistanceCPP(city_position, grid, city_num):
    path = Astar.GetPath(grid, city_position)
    distance = np.zeros((city_num,city_num),dtype=np.int32)
    for i in range(0,city_num):
        for j in range(0,city_num):
            distance[i][j] = len(path[i][j])
    return path,distance

