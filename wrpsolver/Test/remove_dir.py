import os
dirpath = '/remote-home/ums_qipeng/WatchRouteProblem/wrpsolver/Test/optimal_path_l/'
names = os.listdir(dirpath)
for name in names:
    os.remove(dirpath + name)
os.rmdir(dirpath)