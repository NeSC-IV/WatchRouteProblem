import os
import json
import cv2
import glob
import shapely
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv
from stable_baselines3 import PPO
if __name__ == "__main__":

    pointList = [[2.8988437993430676, 1.7058561708349658], [2.8011465610868096, 61.30117150715253], [62.60000000000001, 61.4011695898263], [62.60000000000001, 64.3], [57.9, 64.3], [57.9, 73.38843053574631], [63.93829044362265, 73.28944216781808], [64.16431468263141, 75.77570879691444], [57.9, 75.9777834640961], [57.9, 116.9392502465483], [55.46098852973933, 117.16097856202653], [55.3554461088335, 115.89446951115653], [43.63407491638143, 115.79342320777332], [43.86074975345172, 113.3], [55.3, 113.3], [55.3, 86.60000000000001], [33.997513329351094, 86.60000000000001], [33.902495535756785, 113.3], [37.53925024654828, 113.3], [37.765087752463394, 115.78421256506621], [31.502435119314516, 115.91468449492346], [31.40235074442171, 86.58996265133136], [10.751060401623732, 86.29066858839225], [10.544215821030178, 83.60168904067604], [55.3, 83.50156424385507], [55.3, 64.4], [1.7, 64.4], [1.7, 83.52372188344945], [4.65861442787808, 83.62574307061766], [4.4421446816847405, 86.2233800249377], [1.7, 86.12544628630609], [1.7, 134.41817871230927], [13.16471554174088, 134.71988175288138], [12.938213387991244, 137.2114054441274], [1.7, 137.01249901248156], [1.7, 170.38406708131168], [31.400000000000002, 169.71557833533095], [31.400000000000002, 137.58861864458157], [19.2, 137.38861864458156], [19.2, 134.6877202348464], [54.3842285357898, 135.29961986155584], [55.3, 135.2423841450427], [55.3, 134.2607497534517], [57.73375022229764, 134.03949973324285], [57.838757947751446, 135.19458471323486], [80.86347744123381, 135.00028328291012], [85.43640297879725, 134.56057890429824], [85.53996644375596, 132.8], [88.25178344238091, 132.8], [88.3517834423809, 134.20000000000002], [125.50000000000001, 134.20000000000002], [125.50000000000001, 137.39999999999998], [111.5, 137.39999999999998], [111.5, 178.90858863181458], [119.65998225430683, 179.00932915347266], [119.44409301709624, 181.6], [112.2, 181.6], [112.2, 199.19822560487458], [150.3, 199.10176990867208], [150.3, 181.6943549308635], [125.53403876874036, 181.49382083182496], [125.76098642615393, 178.9973966002756], [150.3, 179.0971486879742], [150.3, 137.39505306551976], [136.15792205314628, 137.29475464036477], [136.33996644375597, 134.20000000000002], [150.0, 134.20000000000002], [150.0, 76.5], [112.75324549847639, 76.5], [111.64219120165075, 92.55978483593405], [109.38288384598329, 92.05771653467461], [109.87840416814377, 75.60644183894682], [88.7899301424601, 75.79389494139733], [88.31030872347466, 108.6], [108.92691293265707, 108.6], [109.32920023973361, 98.34167366954824], [111.9202030387265, 98.55759056946434], [111.59799125984686, 108.97577141990675], [111.18423560843365, 111.18246822744385], [88.39999999999999, 111.87892957791996], [88.39999999999999, 115.64409301709622], [85.79235102989557, 115.86139709793824], [86.19296500087495, 75.8], [81.25177820251663, 75.8], [81.04425320543199, 73.1021750378995], [150.5, 72.90201438431286], [150.5, 71.06074975345172], [152.98744468012868, 70.83461841889455], [153.1886475275943, 83.00739069056615], [207.4, 83.59240528559212], [207.4, 61.28009886695574], [186.48763249999212, 60.676857496763205], [186.89945673026338, 58.0], [207.4, 58.0], [207.4, 33.400000000000006], [179.4, 33.400000000000006], [179.4, 58.051778202516644], [180.82755075662627, 58.16158979918019], [180.3746379856927, 60.65261003931507], [176.7976190325748, 60.34156491295699], [176.8974026063265, 33.400000000000006], [153.09778130510577, 33.400000000000006], [152.99626656506229, 65.37714311369751], [151.01982999172264, 64.29908680096683], [91.89999999999999, 63.99642555527543], [91.89999999999999, 61.29879414387403], [149.9988514228084, 61.39879216697697], [149.90115082545728, 1.7037271854441602], [76.09885005933381, 2.0962926150512], [76.00114221404863, 61.209539012580905], [83.24482364420645, 61.308767525322786], [83.45845875710683, 63.87238888012749], [71.2, 64.32640587113143], [71.2, 61.2], [73.39999999999999, 61.2], [73.39999999999999, 2.2941829263341895], [2.8988437993430676, 1.7058561708349658]]
    polygon = shapely.Polygon(pointList)
    env = GridWorldEnv(render=True)
    model = PPO.load('saved_model/decay')
    rewardList = []
    while True:
        shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/")
        shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/")
        shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/")
        os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/")
        os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/")
        os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/")

        # observation,_ = env.reset(polygon=polygon)
        observation,_ = env.reset()
        Done = False
        state = None
        action ,state= model.predict(observation,state,deterministic=True)
        rewardSum = 0
        cnt = 1
        while not Done:
            action = int(action)
            observation,reward,Done,_,_ = env.step(action)
            action ,state = model.predict(observation,state,deterministic=True)
            print(action,reward,cnt)
            rewardSum += reward
            cnt += 1

        # fig, ax = plt.subplots()
        # ims = []
        # for pic in range(cnt):
        #     im = ax.imshow(plt.imread("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/" + str(pic)+".png"), animated = True)
        #     ims.append([im])
        # ani = animation.ArtistAnimation(fig, ims, interval=60)
        # ani.save('/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result.gif')

        frames = []
        for pic in range(cnt):
            im = Image.open("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/" + str(pic)+".png")
            frames.append(im)
        frame_one = frames[0]
        frame_one.save("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result.gif", format="GIF", append_images=frames,
               save_all=True, duration=60, loop=0)
        
        frames = []
        for pic in range(cnt):
            im = Image.open("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp3/" + str(pic)+".png")
            frames.append(im)
        frame_one = frames[0]
        frame_one.save("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result1.gif", format="GIF", append_images=frames,
               save_all=True, duration=60, loop=0)
        
        frames = []
        for pic in range(cnt):
            im = Image.open("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/" + str(pic)+".png")
            frames.append(im)
        frame_one = frames[0]
        frame_one.save("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/result2.gif", format="GIF", append_images=frames,
               save_all=True, duration=60, loop=0)

        print("gif saved!")
        rewardList.append(rewardSum)
        cmd = input()
        if cmd == 'q':
            break

    print('reward: ',rewardList)