import os
import json
import cv2
import glob
import shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from wrpsolver.bc.gym_env_hwc_100_pos import GridWorldEnv
from stable_baselines3 import PPO
if __name__ == "__main__":


    env = GridWorldEnv(render=True)
    model = PPO.load('saved_model/test')
    rewardList = []
    while True:
        shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/")
        shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/")
        shutil.rmtree("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/")
        os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp/")
        os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp1/")
        os.mkdir("/remote-home/ums_qipeng/WatchRouteProblem/render_saved/tmp2/")

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

        print("gif saved!")
        rewardList.append(rewardSum)
        cmd = input()
        if cmd == 'q':
            break

    print('reward: ',rewardList)