import os
import cv2
import json
import shapely
import numpy as np

def GetPolygon(seed):
    map_file = os.path.dirname(os.path.abspath(__file__))+"/maps/"+str(seed)+'.png'
    img = cv2.imread(map_file)
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # grayImg = cv2.resize(grayImg,(320,240),interpolation = cv2.INTER_NEAREST)
    ret,binImg = cv2.threshold(grayImg,128,255,0)
    contours, hierarchy	=cv2.findContours(binImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygonPoint = np.reshape(contours[0],(-1,2)).tolist()
    holeList = []
    for i in range(1,len(contours)):
        holePoint = np.reshape(contours[i],(-1,2)).tolist()
        holeList.append(holePoint)
    polygon = shapely.Polygon(shell=polygonPoint,holes=holeList)
    polygon = polygon.simplify(0.05,preserve_topology=True)
    return polygon

if __name__ == '__main__':
    import os
    path=map_file = os.path.dirname(os.path.abspath(__file__))+"/maps/"     

    #获取该目录下所有文件，存入列表中
    fileList=os.listdir(path)
    fileList.sort()

    n=0
    for i in fileList:
        
        #设置旧文件名（就是路径+文件名）
        oldname=path + fileList[n]   # os.sep添加系统分隔符
        
        #设置新文件名
        newname=path + str(n) +'.png'
        
        os.rename(oldname,newname)   #用os模块中的rename方法对文件改名
        print(oldname,'======>',newname)
        
        n+=1
    print(n)
