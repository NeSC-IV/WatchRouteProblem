from wrpsolver.Test.save_trajectory import GetTrajectory
import sys
if __name__ =='__main__':
    if len(sys.argv) == 2:
        seed = sys.argv[1]
    else:
        seed = 1
    GetTrajectory(int(seed))