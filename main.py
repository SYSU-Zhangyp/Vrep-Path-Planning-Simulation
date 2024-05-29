try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import reporter as rp
import Astar1 as As
import smooth as sm
    
class BubbleRobController:
    def __init__(self):
        # PI控制参数
        self.Kp_linear = 0.0001
        self.Ki_linear = 0.00005
        self.Kp_angular = 0.006
        self.Ki_angular = 0.0002
        # self.Kp_angular = 0.01
        # self.Ki_angular = 0.00034

        # 初始化其他参数
        self.prev_error_linear = 0
        self.prev_error_angular = 0
        self.integral_linear = []
        self.integral_angular = []
        self.max_velocity = 4 * math.pi
        self.L = 0.2  # 轮子间距
        self.R = 0.04  # 轮子半径
        self.alpha = 0.4
        self.beta = 0.2

        self.vel_L = 0.0
        self.vel_R = 0.0

    def distance_error(self, position1, position2):
        return math.hypot(position2[0] - position1[0], position2[1] - position1[1])

    def angle_error(self, red_position, blue_position, green_position):
        # 计算蓝色到红色的向量
        BR_vector = np.array([red_position[0] - blue_position[0], red_position[1] - blue_position[1]])
        # 计算红色到绿色的向量
        RG_vector = np.array([green_position[0] - red_position[0], green_position[1] - red_position[1]])

        # 计算两个向量的点积
        dot_product = np.dot(BR_vector, RG_vector)
        # 计算两个向量的叉积
        cross_product = np.cross(BR_vector, RG_vector)

        # 计算夹角的弧度
        angle_radians = np.arctan2(cross_product, dot_product)
        # 将弧度转换为角度
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def control_bubble(self, current_position, target_position, red_position, blue_position):
        error_linear = self.distance_error(target_position, current_position)
        error_angular = self.angle_error(red_position, blue_position, target_position)

        print("~~~~~~")
        # 限制误差防止抖动 0.1
        if error_linear < 0.1:
            error_linear = 0
        if abs(error_angular) < 5:
            error_angular = 0

        # 计算线速度控制量
        if len(self.integral_linear) < 30:
            self.integral_linear.append(error_linear)
        else:
            self.integral_linear.pop(0)
            self.integral_linear.append(error_linear)
        integral_linear_sum = sum(self.integral_linear)
        linear_velocity = self.Kp_linear * error_linear + self.Ki_linear * integral_linear_sum

        # 计算角速度控制量
        if len(self.integral_angular) < 30:
            self.integral_angular.append(error_angular)
        else:
            self.integral_angular.pop(0)
            self.integral_angular.append(error_angular)
        integral_angular_sum = sum(self.integral_angular)
        angular_velocity = self.Kp_angular * error_angular + self.Ki_angular * integral_angular_sum

        print("linear_velocity=", linear_velocity)
        print("angular_velocity=", angular_velocity)

        # 计算左右轮速度
        vl = linear_velocity + self.L * 0.5 * angular_velocity + 0.15
        vr = linear_velocity - self.L * 0.5 * angular_velocity + 0.15

        # vl = self.correct(vl, self.vel_L)
        # vr = self.correct(vr, self.vel_R)         

        if abs(vl-vr) > self.beta:
            if vl > vr:
                vl = vr + self.beta
            else:
                vl = vr - self.beta 

        # 限制速度
        # vl = max(min(vl, self.max_velocity), -self.max_velocity)
        # vr = max(min(vr, self.max_velocity), -self.max_velocity)

        # 设置左右轮速度
        sim.simxSetJointTargetVelocity(self.clientID, self.left_jointHandle, vl / self.R, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(self.clientID, self.right_jointHandle, vr / self.R, sim.simx_opmode_streaming)

    def correct(self, v, ori_v):
        if abs(v-ori_v) < self.alpha:
            ori_v = v
        elif ori_v > v:
            v = ori_v - self.alpha
            ori_v = v
        else:
            v = ori_v + self.alpha
            ori_v = v     
        return v

    def connect_to_simulator(self):
        # 尝试关闭所有的已开启的连接，开始运行program
        print ('Program started')
        sim.simxFinish(-1) # just in case, close all opened connections

        # 连接到CoppeliaSim的远程API服务器
        self.clientID = sim.simxStart('127.0.0.1', -3, True, True, 5000, 5)
        if self.clientID != -1:
            print('Connected to remote API server')
        else:
            print('Failed connecting to remote API server')
            sys.exit(1)

        # 设置同步模式
        synchronous = True
        sim.simxSynchronous(self.clientID, synchronous)
        time.sleep(2)

    def start_simulation(self):
        # 开始仿真
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_streaming)

    def stop_simulation(self):
        # 停止仿真
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_streaming)

    def disconnect_from_simulator(self):
        # 关闭连接
        sim.simxFinish(self.clientID)

    def __del__(self):
        self.disconnect_from_simulator()

    def Astar(self, image_data, lower_white, upper_white, center, destination):
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        
        # 创建白色区域的掩码
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # 将图像中的白色区域转换为1，其他区域转换为0
        binary_map = mask_white.astype(int)

        # 直接将掩码图像作为二值图像使用
        binary_map = mask_white

        # 直接将掩码图像作为二值图像使用
        binary_map = np.array(mask_white)
        

        # save_map_to_file(binary_map, "origin.csv")
        # 膨胀操作，将白色区域扩展一定距离
        kernel = np.ones((40, 60), np.uint8)  # 定义膨胀核的大小
        binary_map = (binary_map / 200).astype(int)
        map1 = As.Map(512, 512, binary_map)

        binary_map = cv2.dilate(binary_map.astype(np.uint8), kernel, iterations=1)      

        # 保存地图信息到文件
        # save_map_to_file(binary_map, "temp1.csv")

        map = As.Map(512, 512, binary_map)
        
        center = (int(center[0]), int(center[1]))
        destination = (int(destination[0]), int(destination[1]))
        path = As.AStarSearch(map, center, destination)
        if path:
            print("路径已找到：", path)
            As.visualize_path(map, path)
        else:
            print("没有找到路径。")

        return path, map1

    def run(self, reporter):
        # 连接到远程 API
        self.connect_to_simulator()
        # 开始启动仿真
        self.start_simulation()

        true_path = []
        # 获取视觉传感器和关节的句柄
        Vision_sensor = 'top_view_camera'
        error, Vision_sensor_Handle = sim.simxGetObjectHandle(self.clientID, Vision_sensor, sim.simx_opmode_blocking)
        if error != sim.simx_return_ok:
            print(f'Failed to get handle for sensor {Vision_sensor}')
            sys.exit(1)

        jointName1 = 'bubbleRob_leftMotor'
        error1, self.left_jointHandle = sim.simxGetObjectHandle(self.clientID, jointName1, sim.simx_opmode_blocking)
        if error1 != sim.simx_return_ok:
            print(f'Failed to get handle for sensor {jointName1}')
            sys.exit(1)

        jointName2 = 'bubbleRob_rightMotor'
        error2, self.right_jointHandle = sim.simxGetObjectHandle(self.clientID, jointName2, sim.simx_opmode_blocking)
        if error2 != sim.simx_return_ok:
            print(f'Failed to get handle for sensor {jointName2}')
            sys.exit(1)

        path = []
        count = 0
        # 主循环
        for i in range(1, 2000):
            print('--------------------------epoch', i, '------------------------------')
            error_code, resolution, image_data = sim.simxGetVisionSensorImage(self.clientID, Vision_sensor_Handle, options=0, operationMode=sim.simx_opmode_streaming) 

            if error_code == sim.simx_return_ok:  
                image_data = np.array(image_data, dtype=np.int16)+256
                image_data = np.array(image_data, dtype=np.uint8)
                image_data.resize([resolution[1], resolution[0], 3])
                image_data = np.flipud(image_data)
                if reporter._init_image is None:
                    reporter.log_init_image(image_data)

                # 转换到HSV颜色空间  
                hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)  
                
                # 定义红色在HSV空间中的范围
                lower_red = np.array([0, 70, 50])  
                upper_red = np.array([10, 255, 255])

                # 定义蓝色在HSV空间中的范围
                lower_blue = np.array([100, 70, 50])  
                upper_blue = np.array([130, 255, 255]) 

                # 定义绿色在HSV空间中的范围
                lower_green = np.array([40, 70, 50])   
                upper_green = np.array([80, 255, 255])

                # 定义白色在HSV空间中的范围
                lower_white = np.array([0, 0, 230])
                upper_white = np.array([180, 10, 255])

                # 创建红色、蓝色、绿色和白色区域的掩码  
                mask_red = cv2.inRange(hsv, lower_red, upper_red)  
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
                mask_green = cv2.inRange(hsv, lower_green, upper_green) 
                mask_white = cv2.inRange(hsv, lower_white, upper_white)

                # 找到红色、蓝色和绿色的区域的轮廓  
                contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
                contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
                contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 计算红色和蓝色区域的中心点  
                if contours_red:  
                    c_red = max(contours_red, key=cv2.contourArea)  
                    M_red = cv2.moments(c_red)  
                    if M_red["m00"] != 0:  
                        cX_red = int(M_red["m10"] / M_red["m00"])  
                        cY_red = int(M_red["m01"] / M_red["m00"])  
                        print(f"Red region center at step {i}: ({cX_red}, {cY_red})")  
                else:  
                    # 没有找到红色轮廓  
                    cX_red, cY_red = None, None  
                
                if contours_blue:  
                    c_blue = max(contours_blue, key=cv2.contourArea)  
                    M_blue = cv2.moments(c_blue)  
                    if M_blue["m00"] != 0:  
                        cX_blue = int(M_blue["m10"] / M_blue["m00"])  
                        cY_blue = int(M_blue["m01"] / M_blue["m00"])  
                        print(f"Blue region center at step {i}: ({cX_blue}, {cY_blue})") 
                else:  
                    # 没有找到蓝色轮廓  
                    cX_blue, cY_blue = None, None  
                
                # 计算绿色区域的中心点（假设障碍物是最大的绿色轮廓）  
                if contours_green:  
                    c_green = max(contours_green, key=cv2.contourArea) 
                    M_green = cv2.moments(c_green)   
                    if M_green["m00"] != 0:  
                        cX_green = int(M_green["m10"] / M_green["m00"])  
                        cY_green = int(M_green["m01"] / M_green["m00"])  
                        print(f"Green region center at step {i}: ({cX_green}, {cY_green})") 
                else:
                    # 没有找到绿色轮廓
                    cX_green, cY_green = None, None  

                # 处理每个白色区域的轮廓
                for contour in contours_white:
                    # 计算白色区域的中心点
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # 在这里进行进一步的处理，例如，将中心点添加到一个列表中，以便后续使用
                        print(f"White region center at step {i}: ({cX}, {cY})")

                # 在图像中绘制白色区域的轮廓
                image_data_cv = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)  # 将图像转换为 BGR 格式
                cv2.drawContours(image_data_cv, contours_white, -1, (0, 255, 0), 2)  # 在转换后的图像上绘制轮廓
                image_data_cv_rgb = cv2.cvtColor(image_data_cv, cv2.COLOR_BGR2RGB)  # 将图像重新转换为 RGB 格式

                # # 在图像中显示结果
                # plt.cla()
                # plt.imshow(image_data_cv_rgb)
                # plt.pause(0.01)                

                # 计算物体中心（红色与蓝色中心的平均值）  
                car_Xcenter = (cX_red + cX_blue) / 2 if cX_red is not None and cX_blue is not None else None  
                car_Ycenter = (cY_red + cY_blue) / 2 if cY_red is not None and cY_blue is not None else None  

                if car_Xcenter is not None and car_Ycenter is not None:
                    if count % 5==0:
                        true_path.append([car_Xcenter, car_Ycenter])

                if (reporter._start_position is None or reporter._goal_position is None) and car_Xcenter is not None and car_Ycenter is not None:
                    car_position = np.array([car_Ycenter, car_Xcenter])
                    reporter.log_start_position(car_position)
                    goal_position = np.array([cY_green, cX_green])
                    reporter.log_goal_position(goal_position)
                    mask_white_inverted = cv2.bitwise_not(mask_white)
                    reporter.log_obstacle_mask(mask_white_inverted)

                if path == [] and cX_green is not None and car_Xcenter is not None:
                    path, map = self.Astar(image_data, lower_white, lower_white, [car_Xcenter, car_Ycenter], [cX_green, cY_green])
                    
                    path = sm.smooth(path)
                    path = sm.smooth(path)
                    # np_path = np.array(path)
                    # np_path_corrected = np_path[:, [1, 0]]
                    # reporter.log_plan_path(np_path_corrected)
                    # reporter.report_plan()
                    As.visualize_path(map, path)
                    aim = [path[i] for i in range(45, len(path) - 90, 45)]
                    aim = sm.smooth(aim)
                    aim.append([cX_green, cY_green])
                    As.visualize_path(map, aim)
                    (target_Xcenter, target_Ycenter) = aim.pop(0)
                    
                if cX_green is not None and car_Xcenter is not None:
                    if near((car_Xcenter, car_Ycenter),(target_Xcenter, target_Ycenter)):
                            if len(aim) == 0:
                                print("到达目的地")        
                                np_path = np.array(true_path)
                                print(np_path)
                                np_path_corrected = np_path[:, [1, 0]]
                                reporter.log_plan_path(np_path_corrected)
                                reporter.report_plan()
                                break
                            (target_Xcenter, target_Ycenter) = aim.pop(0)
                    
                    # print("======================================", (target_Xcenter, target_Ycenter))
                    self.control_bubble((car_Xcenter, car_Ycenter), (target_Xcenter, target_Ycenter), (cX_red, cY_red), (cX_blue, cY_blue))       

                sim.simxSynchronousTrigger(self.clientID)
        
        self.stop_simulation()
        print('Program ended')



def near(car_center, target_center):
    (car_Xcenter, car_Ycenter) = car_center
    (target_Xcenter, target_Ycenter) = target_center
    distance = np.sqrt((car_Xcenter-target_Xcenter)**2+(car_Ycenter-target_Ycenter)**2)
    
    if distance > 8:
        return False
    else:
        return True

def save_map_to_file(binary_map, filename):
    with open(filename, 'w') as file:
        for row in binary_map:
            file.write(','.join(map(str, row)) + '\n')


if __name__ == "__main__":
    # 实例化控制器并运行
    controller = BubbleRobController()
    reporter = rp.Reporter()
    
    controller.run(reporter)
