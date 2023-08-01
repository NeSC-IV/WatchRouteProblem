from wrpsolver.bc.gym_env_hwc_100_pp_dict import GridWorldEnv
import shapely
import cv2
import sys
import tty
import termios
import numpy as np
from wrpsolver.Test.draw_pictures import *
import time
def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch
def key2move(key):
    actionDict = {'d':0,'a':1,'s':2,'w':3,'x':4,'q':5,'z':6,'e':7}
    return actionDict[key]
pointList = [[1.2, 8.4], [1.2, 33.3], [36.1, 33.3], [36.1, 31.6], [36.18265511495615, 31.270022284221803], [36.35592470108545, 31.058892682646086], [36.59680072592188, 30.930141764987457], [37.4, 30.9], [37.84407529891455, 31.058892682646086], [38.16985823501254, 31.49680072592188], [38.2, 33.3], [53.3, 33.3], [53.56787840265556, 33.3532843272421], [53.79497474683059, 33.50502525316942], [53.9467156727579, 33.73212159734444], [54.0, 34.0], [53.996629308670535, 34.668611998230695], [53.917344885043846, 34.9299777157782], [53.58889916311373, 35.28202872861178], [53.2, 35.4], [21.5, 35.4], [21.49662930867054, 40.168611998230695], [21.294974746830583, 40.59497474683058], [21.067878402655563, 40.7467156727579], [20.8, 40.8], [19.832121597344436, 40.7467156727579], [19.605025253169416, 40.59497474683058], [19.430141764987454, 40.303199274078125], [19.4, 35.4], [1.2, 35.4], [1.2, 49.8], [19.4, 49.8], [19.4, 46.0], [19.482655114956152, 45.6700222842218], [19.655924701085446, 45.45889268264609], [19.896800725921874, 45.33014176498746], [20.83656322541129, 45.313450303717744], [21.088899163113723, 45.41797127138822], [21.341107317353917, 45.65592470108545], [21.5, 46.1], [21.49662930867054, 51.168611998230695], [21.341107317353913, 51.54407529891455], [20.903199274078123, 51.86985823501254], [11.2, 51.9], [11.169858235012546, 52.20319927407812], [11.041107317353914, 52.44407529891455], [10.729977715778196, 52.71734488504385], [10.4, 52.8], [9.731388001769307, 52.79662930867054], [9.411100836886279, 52.68202872861178], [9.1532843272421, 52.36787840265556], [9.1, 51.9], [1.2, 51.9], [1.2, 74.4], [9.1, 74.4], [9.1, 58.0], [9.1532843272421, 57.73212159734444], [9.305025253169417, 57.505025253169414], [9.532121597344437, 57.3532843272421], [9.8, 57.3], [10.767878402655564, 57.3532843272421], [11.041107317353916, 57.55592470108545], [11.196629308670538, 57.931388001769314], [11.2, 76.7], [15.841084091793707, 76.7], [16.2, 76.6], [20.3, 76.6], [20.658915908206293, 76.7], [24.041084091793707, 76.7], [24.4, 76.6], [31.34108409179371, 76.6], [31.7, 76.5], [33.9, 76.5], [33.903370691329464, 75.73138800176932], [34.0, 75.44108409179371], [34.0, 58.8], [34.0532843272421, 58.53212159734444], [34.20502525316942, 58.30502525316941], [34.4700222842218, 58.082655114956154], [34.8, 58.0], [43.1, 58.0], [43.45891590820629, 58.1], [57.0, 58.1], [57.38889916311372, 58.217971271388215], [57.68202872861178, 58.511100836886285], [57.8, 58.9], [57.8, 74.4], [59.74108409179371, 74.4], [60.1, 74.3], [74.14108409179372, 74.3], [74.5, 74.2], [81.24108409179371, 74.2], [81.6, 74.1], [91.0, 74.1], [91.0, 35.3], [88.5, 35.3], [88.4467156727579, 36.46787840265556], [88.29497474683059, 36.69497474683058], [88.06787840265557, 36.846715672757895], [87.4, 36.9], [87.36985823501254, 38.90319927407812], [87.09497474683059, 39.29497474683058], [86.73656322541129, 39.48654969628226], [71.2, 39.5], [70.81110083688628, 39.382028728611786], [70.5532843272421, 39.06787840265556], [70.5, 36.9], [69.93212159734443, 36.846715672757895], [69.51797127138822, 36.488899163113715], [69.40337069132946, 36.168611998230695], [69.4, 35.4], [59.13138800176931, 35.39662930867054], [58.81110083688628, 35.282028728611785], [58.5532843272421, 34.96787840265556], [58.503370691329465, 33.93138800176931], [58.70502525316941, 33.50502525316942], [58.93212159734444, 33.3532843272421], [59.2, 33.3], [75.7, 33.3], [75.7532843272421, 33.03212159734444], [75.90502525316941, 32.80502525316942], [76.13212159734444, 32.653284327242105], [76.4, 32.6], [77.0686119982307, 32.60337069132946], [77.3299777157782, 32.68265511495615], [77.64110731735391, 32.95592470108545], [77.79197333843123, 33.3], [92.0410840917937, 33.3], [92.4, 33.2], [98.0, 33.2], [98.0, 8.3], [77.9, 8.3], [77.9, 12.7], [77.8, 13.058915908206293], [77.8, 27.4], [77.7467156727579, 27.66787840265556], [77.59497474683059, 27.89497474683058], [77.36787840265556, 28.046715672757898], [77.1, 28.1], [76.13212159734444, 28.046715672757898], [75.90502525316941, 27.89497474683058], [75.73014176498745, 27.603199274078122], [75.7, 23.8], [70.13138800176931, 23.79662930867054], [69.70502525316941, 23.594974746830584], [69.48265511495615, 23.329977715778195], [69.40337069132946, 23.068611998230693], [69.4532843272421, 22.132121597344437], [69.65592470108545, 21.858892682646083], [69.9634367745887, 21.71345030371774], [75.7, 21.7], [75.7, 1.2], [62.4, 1.2], [62.4, 21.7], [64.26861199823068, 21.70337069132946], [64.52997771577819, 21.78265511495615], [64.74110731735392, 21.955924701085447], [64.89662930867054, 22.331388001769305], [64.86985823501254, 23.303199274078125], [64.64407529891454, 23.641107317353917], [64.2, 23.8], [60.33138800176931, 23.79662930867054], [59.90502525316941, 23.594974746830584], [59.61345030371774, 23.136563225411287], [59.60337069132946, 22.331388001769305], [59.68265511495615, 22.0700222842218], [59.911100836886284, 21.817971271388217], [60.3, 21.7], [60.3, 1.4], [38.2, 1.4], [38.2, 21.7], [54.468611998230685, 21.70337069132946], [54.78889916311372, 21.81797127138822], [55.01734488504385, 22.0700222842218], [55.096629308670536, 22.331388001769305], [55.1, 23.0], [55.0467156727579, 23.267878402655562], [54.894974746830584, 23.494974746830586], [54.6299777157782, 23.71734488504385], [54.3, 23.8], [38.2, 23.8], [38.186549696282256, 25.83656322541129], [37.99497474683058, 26.19497474683058], [37.56861199823069, 26.396629308670537], [36.53212159734444, 26.3467156727579], [36.217971271388215, 26.088899163113723], [36.10337069132946, 25.768611998230693], [36.1, 8.4], [1.2, 8.4]]
actionList = [0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 7, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 0, 0, 0, 7, 0, 7, 0, 7, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 7, 7, 7, 0, 0, 0, 3, 3, 3, 3, 5, 3, 3, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 0, 1, 1, 5, 5, 5, 1, 5, 1, 5, 1, 5, 5, 2, 6, 6, 1, 1, 1, 1, 5, 3, 5, 6, 6, 6, 1, 1, 6, 1, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 5, 1, 5, 5]
startPoint = (9, 56)
# pointList = [[40, 1], [40, 1], [40, 61], [84, 61], [84, 60], [84, 60], [85, 60], [85, 60], [86, 60], [86, 60], [86, 61], [86, 61], [86, 61], [86, 70], [86, 70], [84, 70], [84, 70], [84, 62], [38, 62], [38, 63], [1, 63], [1, 63], [1, 149], [55, 149], [55, 120], [55, 120], [55, 91], [55, 91], [55, 90], [56, 89], [57, 89], [57, 89], [84, 89], [84, 81], [84, 81], [86, 81], [86, 81], [86, 91], [86, 91], [86, 91], [86, 91], [85, 91], [85, 109], [85, 109], [85, 127], [85, 127], [85, 134], [87, 134], [87, 134], [87, 136], [87, 136], [85, 136], [85, 164], [85, 164], [85, 196], [165, 196], [165, 136], [98, 136], [98, 136], [98, 134], [98, 134], [165, 134], [165, 134], [165, 129], [166, 129], [166, 122], [166, 122], [166, 115], [166, 115], [166, 108], [166, 108], [166, 101], [166, 101], [166, 94], [166, 94], [166, 87], [166, 87], [166, 80], [167, 80], [167, 73], [167, 73], [167, 66], [167, 66], [167, 59], [167, 59], [167, 57], [125, 57], [125, 57], [125, 55], [125, 55], [169, 55], [169, 55], [199, 55], [199, 1], [115, 1], [115, 48], [115, 48], [114, 48], [114, 48], [114, 1], [86, 1], [86, 14], [86, 14], [86, 37], [86, 37], [86, 49], [86, 49], [84, 49], [84, 49], [84, 41], [84, 41], [84, 1]]
# startPoint = (55, 65)
# actionList = [1,1,0,0,0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 6, 2, 6, 6, 2, 2, 2, 2, 6, 6, 6, 6, 2, 2, 2, 6, 2, 2, 2, 6, 6, 2, 6, 2, 6, 2, 6, 2, 6, 6, 2, 2, 6, 2, 2, 6, 2, 6, 2, 6, 2]
testList = []
def main():
    polygon = shapely.Polygon(pointList)
    gridEnv = GridWorldEnv(channel=True,render=True)
    # gridEnv = GridWorldEnv(polygon,startPoint)
    for i in range(1):
        gridEnv.reset()
        rewardSum = 0
        Done = False
        cnt = 0

        for action in actionList[:]:
        # while not Done:
        #     key = readchar()
        #     action = key2move(key)
            testList.append(gridEnv.pos)
            observation,reward,Done,_,_ = gridEnv.step(action)
            print(reward,cnt)
            rewardSum += reward
            # temp = np.transpose(observation,(2,0,1))
            # cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp/' + str(cnt) +'.png',observation["image"])
            cnt += 1
            if(Done):
                break
        time.sleep(2)
        print(rewardSum)
if __name__ == '__main__':
    while(True):
        main()
        break