import os
import json
import numpy as np
import math
from ..Global import *

meter2pixel = zoomRate

def GetPolygon(seed):
    json_path = os.path.dirname(os.path.abspath(__file__))+"/json/"
    map_file = os.path.dirname(os.path.abspath(__file__))+"/map_id_35000.txt"
    map_ids = np.loadtxt(map_file, str)

    file_name = map_ids[seed]
    print(file_name)
    with open(json_path + '/' + file_name + '.json') as json_file:
        json_data = json.load(json_file)


    bbox = json_data['bbox']
    maxNum = max(bbox['max'][0],bbox['max'][1])
    verts = (np.array(json_data['verts']) * meter2pixel / math.ceil(maxNum)).astype(int)
    return verts,file_name

    


if __name__ == '__main__':

    json_path = os.path.dirname(os.path.abspath(__file__))+"/json/"
    map_file = os.path.dirname(os.path.abspath(__file__))+"/map_id_10000.txt"
    # map_file = os.path.abspath(os.path.join(os.getcwd(), "./map_id_1000.txt"))
    print("---------------------------------------------------------------------")
    print("|map id set file path        |{}".format(map_file))
    print("---------------------------------------------------------------------")
    print("|json file path              |{}".format(json_path))
    print("---------------------------------------------------------------------")

    map_ids = np.loadtxt(map_file, str)

    for map_id in map_ids:
        draw_map(map_id, json_path)
        break

