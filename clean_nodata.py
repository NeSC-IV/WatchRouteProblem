import os
import json
import shutil
import numpy as np
path = "/remote-home/ums_qipeng/WatchRouteProblem/wrpsolver/Test/pic_data_picsize100_pos/"

def main():
    filesNames = os.listdir(path)
    cnt = 1
    for fileName in filesNames[:]:
        dirPath = path + fileName
        files = os.listdir(dirPath)
        if 'data.json' not in files:
            shutil.rmtree(dirPath)
            print(cnt)
            cnt += 1
if __name__ =='__main__':
    main()

