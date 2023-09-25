import numpy as np
import astar
# a = np.array([[1,2,3],[4,5,6]])
a = np.zeros((10,10),dtype=np.uint8)
a.fill(1)
cityPos = np.array([[1,2],[3,4],[5,6]])
goods_class = np.array([0,0,0,0,0])
# a = astar.sum_2d(a,a,goods_class)
res = astar.sum_2d(a,cityPos,goods_class)
for i in range(len(res)):
    print(res[i])