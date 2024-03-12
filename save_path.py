import os

# @func_set_timeout(30)
def save_path(begin,end):
    return os.system("timeout 2h /root/anaconda3/bin/python /remote-home/ums_qipeng/WatchRouteProblem/run_save_optimal_paths.py "
                            + str(begin) + " " + str(end))

if __name__ == '__main__':
    begin=19933
    goal=30000
    step=250
    for i in range(begin,goal,step):
        success = save_path(i,i+step)
    
