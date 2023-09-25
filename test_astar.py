import numpy as np
import astar
a = np.array([[1,2,3],[4,5,6]])
cityPos = np.array([[1,2],[3,4],[5,6]])
goods_class = np.array([0,0,0,0,0])
# a = astar.sum_2d(a,a,goods_class)
res = astar.sum_2d(a,cityPos,goods_class)
print(res)