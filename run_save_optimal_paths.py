from wrpsolver.Test.save_optimal_path import SaveOptimalPath
import sys
if __name__ =='__main__':
    begin = 0
    end = 10000
    if(len(sys.argv) == 3):
        begin = int(sys.argv[1])
        end = int(sys.argv[2])
    SaveOptimalPath(begin,end)