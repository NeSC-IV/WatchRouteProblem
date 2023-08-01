import re
import matplotlib.pyplot as plt
import numpy as np
def GetSB3Log():
    file = "ppo_pp_skip_stack.log"
    re_rew = r'\-?\d+\.?\d*'
    re_loss = r'\-?\d+\.?\d*'
    re_step = r'\d+'
    rewardList = []
    stepList = []
    lossList = []
    with open(file) as f:
        while True:
            l = f.readline()
            if "ep_rew_mean" in l:
                result = re.search(re_rew,l)
                reward = float(result.group())
                rewardList.append(reward)
            if "total_timesteps" in l:
                result = re.search(re_step,l)
                step = int(result.group())
                stepList.append(step)
            if "entropy_loss" in l:
                result = re.search(re_loss,l)
                loss = -float(result.group())
                lossList.append(loss)
            if not l :
                break
    return rewardList[1:],lossList,stepList[1:]
def plot(x,y,path,label):
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') 

    plt.figure()                   # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
    plt.plot(x,y,'b',label = label) 
    plt.ylabel(label)
    plt.xlabel('steps')
    plt.legend()        #个性化图例（颜色、形状等）
    plt.savefig(path) #保存图片 路径：/imgPath/
def main():
    rewardList,lossList,stepList = GetSB3Log()
    
    plot(stepList,rewardList,"/remote-home/ums_qipeng/WatchRouteProblem/reward.jpg","reward")
    plot(stepList,lossList,"/remote-home/ums_qipeng/WatchRouteProblem/Loss.jpg","Loss")

main()