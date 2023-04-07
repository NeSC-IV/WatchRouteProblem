import shapely
import os
import cv2
import json
from wrpsolver.bc.bc import *
from random import choice,shuffle
paddle.set_device("gpu")
state_dict = paddle.load('/home/nianba/polycy.pdparams')
bc = BehaviorClone(1,8,1e-5)
bc.policy.set_state_dict(state_dict)
dirPath = os.path.dirname(os.path.abspath(__file__))+"/pic_data/"
picDirNames = os.listdir(dirPath)
picDataDirs = [(dirPath+picDirName )for picDirName in picDirNames]
shuffle(picDataDirs)
cnt = 0
rewardList = []
try:
    for picDataDir in picDataDirs:
        print('第 ' + str(cnt) +' 次训练开始：')
        filesNames = os.listdir(picDataDir)
        filesNames.remove('data.json')
        picIDs = [int(fileName.split('.')[0]) for fileName in filesNames]
        picIDs.sort()
        picDirs = [picDataDir+'/'+str(picID)+'.png' for picID in picIDs]
        dataDir = picDataDir+'/data.json'
        pics = [cv2.imread(picDir,cv2.IMREAD_GRAYSCALE) for picDir in picDirs]
        with open(dataDir) as json_file:
            json_data = json.load(json_file)
        picTensorList = [paddle.to_tensor(pic,paddle.uint8) for pic in pics]
        actionList = json_data['actionArray']
        for picTensor,action in zip(picTensorList,actionList):
            bc.learn(picTensor,action)

        testJsonDir = dirPath + choice(picDirNames) + '/data.json'
        with open(testJsonDir) as json_file:
            json_data = json.load(json_file)
        polygon = shapely.Polygon(json_data['polygon'])
        cnt += 1
        if(cnt % 50 == 0):
            reward = test_agent(bc,polygon)
            print('The reward is: ' + str(reward))
            rewardList.append(reward)
except Exception as e:
    print(e)
finally:
    paddle.save(bc.policy.state_dict(),'/home/nianba/polycy.pdparams')
    
print(rewardList)