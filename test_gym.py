from wrpsolver.bc.gym_env_hwc import GridWorldEnv
from stable_baselines3.common.env_checker import check_env
import shapely
import cv2
from timeit import timeit
# pointList = [[21, 1], [21, 1], [21, 50], [33, 50], [34, 50], [34, 52], [33, 52], [1, 52], [1, 54], [1, 54], [1, 83], [1, 83], [1, 109], [1, 109], [1, 125], [28, 125], [28, 126], [28, 127], [28, 127], [1, 127], [1, 198], [65, 198], [65, 197], [65, 197], [65, 127], [43, 127], [42, 127], [42, 126], [43, 125], [68, 125], [68, 126], [68, 127], [68, 127], [67, 127], [67, 198], [110, 198], [110, 127], [82, 127], [82, 127], [82, 126], [82, 125], [92, 125], [92, 114], [92, 114], [92, 102], [92, 102], [94, 102], [94, 102], [94, 125], [139, 125], [139, 126], [139, 127], [139, 127], [111, 127], [111, 128], [111, 129], [111, 198], [184, 198], [184, 127], [154, 127], [153, 127], [153, 126], [154, 126], [184, 126], [184, 27], [94, 27], [94, 27], [94, 49], [94, 49], [94, 70], [94, 70], [92, 70], [92, 70], [92, 60], [92, 60], [92, 52], [78, 52], [78, 52], [78, 50], [78, 50], [91, 50], [91, 1]]
# actionList = [2, 6, 2, 2, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 4, 4, 2, 4, 2, 2, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2, 4, 3, 3, 3, 7, 7, 7, 3, 3, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2, 4]
# startPoint = (37, 50)
pointList = [[40, 1], [40, 1], [40, 61], [84, 61], [84, 60], [84, 60], [85, 60], [85, 60], [86, 60], [86, 60], [86, 61], [86, 61], [86, 61], [86, 70], [86, 70], [84, 70], [84, 70], [84, 62], [38, 62], [38, 63], [1, 63], [1, 63], [1, 149], [55, 149], [55, 120], [55, 120], [55, 91], [55, 91], [55, 90], [56, 89], [57, 89], [57, 89], [84, 89], [84, 81], [84, 81], [86, 81], [86, 81], [86, 91], [86, 91], [86, 91], [86, 91], [85, 91], [85, 109], [85, 109], [85, 127], [85, 127], [85, 134], [87, 134], [87, 134], [87, 136], [87, 136], [85, 136], [85, 164], [85, 164], [85, 196], [165, 196], [165, 136], [98, 136], [98, 136], [98, 134], [98, 134], [165, 134], [165, 134], [165, 129], [166, 129], [166, 122], [166, 122], [166, 115], [166, 115], [166, 108], [166, 108], [166, 101], [166, 101], [166, 94], [166, 94], [166, 87], [166, 87], [166, 80], [167, 80], [167, 73], [167, 73], [167, 66], [167, 66], [167, 59], [167, 59], [167, 57], [125, 57], [125, 57], [125, 55], [125, 55], [169, 55], [169, 55], [199, 55], [199, 1], [115, 1], [115, 48], [115, 48], [114, 48], [114, 48], [114, 1], [86, 1], [86, 14], [86, 14], [86, 37], [86, 37], [86, 49], [86, 49], [84, 49], [84, 49], [84, 41], [84, 41], [84, 1]]
startPoint = (55, 65)
actionList = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 6, 2, 6, 6, 2, 2, 2, 2, 6, 6, 6, 6, 2, 2, 2, 6, 2, 2, 2, 6, 6, 2, 6, 2, 6, 2, 6, 2, 6, 6, 2, 2, 6, 2, 2, 6, 2, 6, 2, 6, 2]
def main():
    polygon = shapely.Polygon(pointList)
    gridEnv = GridWorldEnv()
    gridEnv.reset(polygon,startPoint)
    # cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp/' + str(11111) +'.png',gridEnv.image)
    # cv2.imshow('aa',image)
    # cv2.waitKey(0)
    rewardSum = 0
    Done = False
    cnt = 0

    for action in actionList:
        _,reward,Done,_ = gridEnv.step(action)
        print(reward)
        rewardSum += reward
        cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp/' + str(cnt) +'.png',gridEnv.image)
        # cv2.imwrite('/remote-home/ums_qipeng/WatchRouteProblem/tmp/pos' + str(cnt) +'.png',gridEnv.observation[2].reshape(200,200))
        if(Done):
            break
        # cv2.imshow('aa',gridEnv.observation)
        # cv2.waitKey(0)
        cnt += 1

    print(rewardSum)
if __name__ == '__main__':
    while(True):
        main()
        break