import os
import json
import pickle
import numpy as np
import shapely
import matplotlib.pyplot as plt
loadArea = True
hist = True
if loadArea:
    with open('area.json','r') as f:
        jsonData = json.load(f)
        ids = jsonData['seeds']
        areas = jsonData['areas']
else:
    json_path = os.path.dirname(os.path.abspath(__file__))+"/json/"
    map_file = os.path.dirname(os.path.abspath(__file__))+"/map_id_35000.txt"
    map_ids = np.loadtxt(map_file, str)
    ids = [i for i in range(0,35000)]
    areas = []
    for seed in ids:
        file_name = map_ids[seed]
        # print(file_name)
        with open(json_path + '/' + file_name + '.json') as json_file:
            json_data = json.load(json_file)
        verts = np.array(json_data['verts'])*1
        p = shapely.Polygon(verts)
        areas.append(p.area)

    jsonData = {'seeds':ids,'areas':areas}
    with open('area.json','w') as f:
        json.dump(jsonData,f)


if hist:
    plt.switch_backend('Agg') 
    plt.figure(figsize=(16,6))                   # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.hist(areas,bins=200,range=(0,1000))
else:
    plt.switch_backend('Agg') 
    plt.figure(figsize=(16,6))                   # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(ids,areas,'b',label = 'area') 
    plt.ylabel('area')
    plt.xlabel('seed')
    # plt.axhline(0.65,linestyle='--',color='red',alpha=0.8,label='BC')
    # plt.xlim(0,1e8)
    plt.ylim(0,2000)
    plt.legend()        #个性化图例（颜色、形状等）
plt.savefig('areacount.png') #保存图片 路径：/imgPath/

cnt = []
for id in ids:
    area = areas[id]
    if area < 300 and area > 80:
        cnt.append(id)
    
print(len(cnt))
with open('seeds_80_300','wb') as f:
    pickle.dump(cnt,f)