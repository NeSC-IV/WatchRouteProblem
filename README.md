# WatchmanRouteProblem
## 一、当前进度
- 完成kernel计算
- 完成visibility计算
- 完成最大凸多边形计算
- TPP问题
- HouseExpo数据集
- 多进程优化速度
## 二、后续规划
- 速度优化
- bugfix
- 接口封装

## 三、依赖安装
```bash
pip install -r requirements.txt
```

## 四、测试方法

1. 简单测试：
```bash
python3 run_test.py 
#如果出现错误，可以尝试重新运行
```
2. 测试指定种子
```bash
python3 run_test.py [number] [1 <= number <= 1000]
```

## 五、文件结构

WRP问题解决步骤：

```c
求出多边形最大凸多边形子集(MACS) -> 遍历凸多边形子集(GTSP)
```

```bash
WatchRouteProblem
├── README.md 
├── requirements.txt #python 依赖库
├── run_test.py #测试运行脚本
└── wrpsolver #主文件夹
    ├── Global #存放全局变量、公用函数等
    │   ├── __init__.py
    ├── GTSP #
    │   ├── astar
    │   ├── gtsp.py
    │   ├── __init__.py
    │   └── samples.py
    ├── __init__.py
    ├── MACS #
    │   ├── compute_kernel.py
    │   ├── compute_visibility.py
    │   ├── __init__.py
    │   ├── polygons_coverage.py
    ├── Test #测试所用文件
    │   ├── draw_pictures.py
    │   ├── __init__.py
    │   ├── json #数据集
    │   ├── map_id_1000.txt #数据集的样本
    │   ├── random_polygons_generate.py #随机生成多边形
    │   ├── test.py #测试文件
    │   ├── test.sh #压力测试脚本
    └── WRP_solver.py #main函数
```


1. c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) compute_visibilty_cpp.cpp -o visibility$(python3-config --extension-suffix) -lCGAL -lgmp -lmpfr
2. c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) AStar.cpp -o Astar$(python3-config --extension-suffix)