# V-rep仿真路径规划
[ *智能机器人 · 个人项目* ],
date[ 2024 年 02 月 – 2024 年 06 月 ]

使用 CoppeliaSim 远程 API 实现小车路径规划算法、PID 控制的仿真

- 基于 V-rep 仿真平台搭建 BubbleRob 小车避障规划场景，使用视觉传感器获取全局信息
- 通过 OpenCV 处理视觉传感器的图像信息，获知地图信息
- 利用获取的地图信息使用 Astar 算法规划最优路径，并使用路径平滑算法
- 远程 API 控制 BubbleRob 关节角速度，基于 PID 控制、满足避障条件准确到达目的地点
