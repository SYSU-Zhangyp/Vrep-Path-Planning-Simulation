"""
Author: Long Cheng
Date: 2024-05-11 15:30:19
LastEditors: Long Cheng
LastEditTime: 2024-05-17 00:05:06
Description: 

Copyright (c) 2024 by longcheng.work@outlook.com, All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os


class Reporter:
    def __init__(self, debug_plot=True):
        self._debug_plot = debug_plot
        # get the last modified time of this file
        this_file = __file__
        last_modified = time.ctime(os.path.getmtime(this_file))
        # log last modified time of this file to the report
        self._report_str = "Reporter last modified: {}\n".format(last_modified)
        self._report_file = "report.txt"
        self._init_image = None
        self._obstacle_mask = None
        self._plan_path = None
        self._start_position = None
        self._goal_position = None
        self._robot_sim_path = np.empty(shape=(0, 2))
        self._robot_sim_time = np.empty(shape=(0, 1))
        self.__goal_tolerance = 5
        self.__inflate_size = 30

    def add(self, message):
        self._report.append(message)

    @staticmethod
    def check_rgb_image(image):
        # check if image is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        # check if image is a 3D array
        if image.ndim != 3:
            raise ValueError("image must be a 3D array")
        # check if image has 3 channels
        if image.shape[2] != 3:
            raise ValueError("image must have 3 channels")
        return True

    @staticmethod
    def check_image_mask(image_mask):
        # check if image_mask is a numpy array
        if not isinstance(image_mask, np.ndarray):
            raise TypeError("image_mask must be a numpy array")
        # check if image_mask is a 2D array
        if image_mask.ndim != 2:
            raise ValueError("image_mask must be a 2D array")
        return True

    @staticmethod
    def check_path(path):
        # check if path is a numpy array
        if not isinstance(path, np.ndarray):
            raise TypeError("path must be a numpy array")
        # check if path is a 2D array
        if path.ndim != 2:
            raise ValueError("path must be a 2D array")
        # check if path has 2 columns
        if path.shape[1] != 2:
            raise ValueError("path must have 2 columns")
        # check if path has at least 2 rows
        if path.shape[0] < 2:
            raise ValueError("path must have at least 2 rows")
        return True

    @staticmethod
    def check_position(position):
        # check if position is a numpy array
        if not isinstance(position, np.ndarray):
            raise TypeError("position must be a numpy array")
        # check if position is a 1D array
        if position.ndim != 1:
            raise ValueError("position must be a 1D array")
        # check if position has 2 elements
        if position.shape[0] != 2:
            raise ValueError("position must have 2 elements")
        return True

    def log_init_image(self, init_image):
        if Reporter.check_rgb_image(init_image):
            self._init_image = init_image
        if self._debug_plot:
            plt.imshow(init_image)
            # set image title
            plt.title("Initial Image, close to continue...")
            plt.show()

    def log_obstacle_mask(self, obstacle_mask):
        if Reporter.check_image_mask(obstacle_mask):
            self._obstacle_mask = obstacle_mask
        if self._debug_plot:
            plt.imshow(obstacle_mask, cmap="gray")
            # set image title
            plt.title("Mask of Obstacles (black), close to continue...")
            plt.show()

    def log_plan_path(self, plan_path):
        if Reporter.check_path(plan_path):
            self._plan_path = plan_path
        if self._debug_plot:
            plt.imshow(self._init_image, cmap="gray")
            # set image title
            plt.title("Plan Path on Initial Image, close to continue...")
            plt.plot(plan_path[:, 1], plan_path[:, 0], "r-")
            plt.show()

    def log_start_position(self, start_position):
        if Reporter.check_position(start_position):
            self._start_position = start_position

    def log_goal_position(self, goal_position):
        if Reporter.check_position(goal_position):
            self._goal_position = goal_position

    def log_robot_sim_position(self, robot_sim_position, sim_time):
        if Reporter.check_position(robot_sim_position):
            self._robot_sim_path = np.insert(
                self._robot_sim_path, self._robot_sim_path.shape[0], values=robot_sim_position, axis=0
            )
        # log simulation time
        if not isinstance(sim_time, (int, float)):
            raise TypeError("sim_time must be a number")
        if sim_time < 0:
            raise ValueError("sim_time must be a non-negative number")
        self._robot_sim_time = np.insert(self._robot_sim_time, self._robot_sim_time.shape[0], values=sim_time, axis=0)

    def check_plan_data(self):
        # check if all required data is logged
        if self._init_image is None:
            raise ValueError("init_image is not logged")
        if self._obstacle_mask is None:
            raise ValueError("obstacle_mask is not logged")
        if self._plan_path is None:
            raise ValueError("plan_path is not logged")
        if self._start_position is None:
            raise ValueError("start_position is not logged")
        if self._goal_position is None:
            raise ValueError("goal_position is not logged")
        return True

    def report_plan(self, plot=True, display=True):
        # check if all required data is logged
        if self.check_plan_data():
            self.calc_paln_path_length()
            self.calc_plan_path_to_obstacle_distance()
            self.check_if_plan_path_properly_smoothed()
            self.save()
            if plot:
                self.plot_plan_path()
            if display:
                self.display_report()

    def report_all(self):
        # check if all required data is logged
        self.report_plan(plot=False, display=False)
        if self._robot_sim_path.size == 0:
            raise ValueError("robot_sim_path is not logged")
        # create report
        self.plot_all()
        self.calc_robot_sim_path_length()
        self.check_goal_error()
        self.save()
        self.display_report()

    def check_goal_error(self):
        goal_error = np.linalg.norm(self._robot_sim_path[-1] - self._goal_position)
        if goal_error > self.__goal_tolerance:
            self._report_str += "Goal offset: {:.2f}\n".format(goal_error)
        else:
            self._report_str += "Goal reached\n"
        return goal_error

    def check_if_plan_path_properly_smoothed(self):
        # check is the plan path is properly smoothed
        # calculate the angle between two consecutive points in the plan path
        angles = np.empty(shape=(self._plan_path.shape[0] - 1, 1))
        for i in range(1, self._plan_path.shape[0]):
            # calculate the angle between two consecutive points
            h_diff = self._plan_path[i, 0] - self._plan_path[i - 1, 0]
            w_diff = self._plan_path[i, 1] - self._plan_path[i - 1, 1]
            angle = np.arctan2(h_diff, w_diff) * 180 / np.pi
            angles[i - 1] = angle
        # calculate the difference between consecutive angles
        angle_diff = np.empty(shape=(angles.shape[0] - 1, 1))
        for i in range(1, angles.shape[0]):
            diff1 = np.abs(angles[i] - angles[i - 1])
            diff2 = np.abs(angles[i] - angles[i - 1] + 360)
            diff3 = np.abs(angles[i] - angles[i - 1] - 360)
            angle_diff[i - 1] = np.min([diff1, diff2, diff3])
        # calculate the average angle difference, skip the first empty element
        avg_angle_diff = np.mean(angle_diff)
        # add to report string
        self._report_str += "Average path point angle difference: {:.2f}\n".format(avg_angle_diff)
        max_angle_diff = np.max(angle_diff)
        # add to report string
        self._report_str += "Max path point angle difference: {:.2f}\n".format(max_angle_diff)
        return avg_angle_diff

    def calc_point_to_obstacle_distance(self, point):
        # calculate distance from point to nearest obstacle
        distance = np.min(np.linalg.norm(point - np.argwhere(self._obstacle_mask == 0), axis=1))
        return distance

    def calc_plan_path_to_obstacle_distance(self):
        # calculate distance from each point in the plan path to the nearest obstacle
        distances = np.empty(shape=(self._plan_path.shape[0], 1))
        for i, point in enumerate(self._plan_path):
            distances[i] = self.calc_point_to_obstacle_distance(point)
        # add to report string
        self._report_str += "Min plan path to obstacle distance: {:.2f}\n".format(np.min(distances))
        # add the average distance to the report string
        self._report_str += "Average plan path to obstacle distance: {:.2f}\n".format(np.mean(distances))
        return distances

    def calc_paln_path_length(self):
        length = np.sum(np.linalg.norm(self._plan_path[1:] - self._plan_path[:-1], axis=1))
        # add to report string
        self._report_str += "Plan path length: {:.2f}\n".format(length)
        return length

    def calc_robot_sim_path_length(self):
        length = np.sum(np.linalg.norm(self._robot_sim_path[1:] - self._robot_sim_path[:-1], axis=1))
        # add to report string
        self._report_str += "Robot path length: {:.2f}\n".format(length)
        return length

    def display_report(self):
        print(self._report_str)

    def save(self):
        with open(self._report_file, "w") as f:
            f.write(self._report_str)

    def inflate_obstacles(self):
        kernel = np.ones((self.__inflate_size, self.__inflate_size), np.uint8)
        inflated_obstacle_mask = cv2.erode(self._obstacle_mask, kernel, iterations=1)
        return inflated_obstacle_mask

    def plot_all(self):
        fig, ax = plt.subplots()
        # ax.imshow(self._obstacle_mask)
        # inflate the obstacles

        ax.imshow(self.inflate_obstacles(), cmap="gray", alpha=0.5)
        ax.plot(self._plan_path[:, 1], self._plan_path[:, 0], "r-")
        ax.plot(self._start_position[1], self._start_position[0], "bo")
        ax.plot(self._goal_position[1], self._goal_position[0], "go")
        ax.plot(self._robot_sim_path[:, 1], self._robot_sim_path[:, 0], "b-")
        # set image title
        plt.title("Plan Path and Robot Path on Inflated Obstacle Image, close to continue...")
        plt.show()
        
    def plot_plan_path(self):
        fig, ax = plt.subplots()
        ax.imshow(self.inflate_obstacles(), cmap="gray", alpha=0.5)
        ax.plot(self._plan_path[:, 1], self._plan_path[:, 0], "r-")
        ax.plot(self._start_position[1], self._start_position[0], "bo")
        ax.plot(self._goal_position[1], self._goal_position[0], "go")
        # set image title
        plt.title("Plan Path on Inflated Obstacle Image, close to continue...")
        plt.show()
        
    
