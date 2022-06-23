# -*- coding: utf-8 -*-

from math import sqrt, pow, atan, asin, sin, pi, cos, tau, remainder
import numpy as np
import random
import gym
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils.renderer import Renderer
import matplotlib.pyplot as plt
from gym.utils import seeding
from gym.error import DependencyNotInstalled
from typing import Optional
import pygame


class DiscreteRobotTranslationEnv(gym.Env):


    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 2
    }


    def __init__(self, render_mode: Optional[str] = None):
        # size of the simulation env
        self.env_x = 600
        self.env_y = 400
        # size of the camera view range
        self.img_width = 320
        self.img_height = 200
        # visualizing
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.viewer_env = rendering.Viewer(self.env_x, self.env_y)
        self.viewer_cam = rendering.Viewer(self.img_width, self.img_height)
        self.view_range = pi / 2  # the view range of the camera on robot (degree)
        self.r = 20  # radius of the target. To simulate yolo, the shape of the target is assumed as a cylinder
        self.r0 = 10 # radius of robot
        self.middle = 0
        # Action space
        self.action_space = spaces.Discrete(3)
        # Observation/State Space
        self.low = np.array([0, 0, -int(self.img_width/2), 0], dtype=int)
        self.high = np.array([self.img_width, self.img_width, self.img_width, self.img_height], dtype=int)
        self.observation_space = spaces.Box(self.low, self.high, dtype=int)
        self.seed()
        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.renderer.reset()
        self.renderer.render_step()
        # robot initial orientation
        self.angle_target = pi
        self.distance = 0

        while self.distance <= (self.r + self.r0 + 100) or self.angle_target >= self.view_range:
            x = random.randint(0, self.env_x)
            y = random.randint(0, self.env_y)
            yaw_deg = random.randint(0, 360)
            yaw = yaw_deg * pi / 180
            self.robot = np.array([x, y, yaw_deg])
            self.distance = random.uniform(self.r + self.r0, np.max([x, y, self.env_x - x, self.env_y - y]) - self.r0)
            self.theta_target = yaw
            xt = x + self.distance * cos(self.theta_target)
            yt = y + self.distance * sin(self.theta_target)
            self.target = np.array([xt, yt])
            # view angle of target object in camera
            self.angle_target = 2 * (asin(self.r / self.distance) % tau)



        # if the part of the target is in range view
        target_up = (self.theta_target + self.angle_target/2) % tau
        target_down = (self.theta_target - self.angle_target/2) % tau
        rob_up = (yaw + pi/4) % tau
        rob_down = (yaw - pi/4) % tau
        if (rob_up - target_up) % tau <= pi / 2:
            in_up = True
            up_bound = target_up
        else:
            in_up = False

        if (target_down - rob_down) % tau <= pi / 2:
            in_down = True
            down_bound = target_down
        else:
            in_down = False

        if not in_up and not in_down:
            up_bound = rob_up
            down_bound = rob_up
        elif (not in_up) and in_down:
            up_bound = rob_up
        elif (not in_down) and in_up:
            down_bound = rob_down

        target_middle = (((up_bound - down_bound)%tau) / 2 + down_bound) % tau
        bbox_middle = int(self.img_width * ((rob_up - target_middle) % tau) / self.view_range)
        bbox_width = int(self.img_width * ((up_bound - down_bound) % tau) / self.view_range)
        bbox_bias = int(self.img_width/2 - bbox_middle)
        bbox_height = int((self.angle_target/self.view_range) * self.img_height)

        # print("Initial step:")
        # print(f"theta_target: {self.theta_target * 180/pi}, yaw: {yaw_deg}, distance: {self.distance}")
        # print(f"Yaw: {yaw_deg}; "
        #       f"The middle of bbox is {bbox_middle}; "
        #       f"Width is {bbox_width}; "
        #       f"Edges are {[up_bound * 180 / pi, down_bound * 180 / pi, target_up * 180 / pi, target_down * 180 / pi, rob_up * 180 / pi, rob_down * 180 / pi]}\n")
        # print(f"The middle of bbox is {bbox_middle}; theta mid is {target_middle}; Edges are {[up_bound, down_bound, target_up, target_down, rob_up, rob_down]}")
        self.state = np.array([bbox_middle, bbox_width, bbox_bias, bbox_height]).astype(int)
        self.last_state = self.state
        self.middle = 0
        # assert self.observation_space.contains(self.state)
        # print(self.state)
        return self.state

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        x, y, yaw_deg = self.robot
        yaw = yaw_deg * pi / 180

        x += 10 * (action - 1) * cos(yaw)
        y += 10 * (action - 1) * sin(yaw)

        self.robot = np.array([x, y, yaw_deg])
        xt, yt = self.target
        self.distance = sqrt(pow(x - xt, 2) + pow(y - yt, 2))
        if xt >= x and yt >= y:
            self.theta_target = asin((yt - y) / self.distance)
        elif xt >= x and yt < y:
            self.theta_target = 2*pi - asin(abs(yt - y) / self.distance)
        elif xt < x and yt >= y:
            self.theta_target = pi - asin(abs(yt - y) / self.distance)
        else:
            self.theta_target = pi + asin(abs(yt - y) / self.distance)
        # view angle of target object in camera
        self.angle_target = 2 * (asin(self.r / self.distance) % tau)
        bbox_height = np.min([int((self.angle_target/self.view_range) * self.img_height), 200])
        # if the part of the target is in range view
        target_up = (self.theta_target + self.angle_target / 2) % tau
        target_down = (self.theta_target - self.angle_target / 2) % tau
        rob_up = (yaw + pi / 4) % tau
        rob_down = (yaw - pi / 4) % tau
        if (rob_up - target_up) % tau <= pi/2:
            in_up = True
            up_bound = target_up
        else:
            in_up = False

        if (target_down - rob_down) % tau <= pi/2:
            in_down = True
            down_bound = target_down
        else:
            in_down = False

        if not in_up and not in_down:
            up_bound = rob_up
            down_bound = rob_up
        elif (not in_up) and in_down:
            up_bound = rob_up
        elif (not in_down) and in_up:
            down_bound = rob_down

        target_middle = (((up_bound - down_bound)%tau) / 2 + down_bound) % tau
        bbox_middle = int(self.img_width * ((rob_up - target_middle) % tau) / self.view_range)
        bbox_width = int(self.img_width * ((up_bound - down_bound) % tau) / self.view_range)
        bbox_bias = int(self.img_width/2 - bbox_middle)
        self.state = np.array([bbox_middle, bbox_width, bbox_bias, bbox_height]).astype(int)


        reward = 0

        self.last_state = self.state

        done = bool(bbox_height > 0.8 * self.img_height)
        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)


    def _render(self, mode='human', close=False):
        [x, y, yaw] = self.state
        [xt, yt] = self.state_t

        env_line_view_m = rendering.Line((x, y), (x + 100 * cos(yaw), y + 100 * sin(yaw)))
        env_line_view_1 = rendering.Line((x, y), (x + 100 * cos(yaw + pi/4), y + 100 * sin(yaw + pi/4)))
        env_line_view_2 = rendering.Line((x, y), (x + 100 * cos(yaw - pi/4), y + 100 * sin(yaw - pi/4)))
        env_line_view_1.set_color(0, 0, 255)
        env_line_view_2.set_color(0, 0, 255)

        env_line_obj_m = rendering.Line((x, y), (xt, yt))
        length = self.distance #* cos(0.5 * self.target_angle)
        env_line_obj_1 = rendering.Line((x, y), (x + length * cos(self.theta_target + 0.5 * self.target_angle),
                                                 y + length * sin(self.theta_target + 0.5 * self.target_angle)))
        env_line_obj_2 = rendering.Line((x, y), (x + length * cos(self.theta_target - 0.5 * self.target_angle),
                                                 y + length * sin(self.theta_target - 0.5 * self.target_angle)))
        env_line_obj_m.set_color(0, 255, 0)
        env_line_obj_1.set_color(0, 255, 255)
        env_line_obj_2.set_color(255, 0, 0)

        self.viewer_env.add_geom(env_line_view_m)
        self.viewer_env.add_geom(env_line_view_1)
        self.viewer_env.add_geom(env_line_view_2)
        self.viewer_env.add_geom(env_line_obj_m)
        self.viewer_env.add_geom(env_line_obj_1)
        self.viewer_env.add_geom(env_line_obj_2)

        env_robot = rendering.make_circle(5)
        # env_robot.set_color(255, 255, 255)
        env_robot_trans = rendering.Transform(translation=(x, y))
        env_robot.add_attr(env_robot_trans)
        env_obj = rendering.make_circle(self.r)
        # env_obj.set_color(255, 255, 255)
        env_obj_trans = rendering.Transform(translation=(xt, yt))
        env_obj.add_attr(env_obj_trans)

        self.viewer_env.add_geom(env_robot)
        self.viewer_env.add_geom(env_obj)

        cam_left = rendering.Line((0, 0), (0, self.img_height))
        cam_right = rendering.Line((self.img_width, 0), (self.img_width, self.img_height))
        cam_up = rendering.Line((0, self.img_height), (self.img_width, self.img_height))
        cam_down = rendering.Line((0, 0), (self.img_width, 0))

        bbox_down = rendering.Line((self.bbox[0], self.bbox[1]), (self.bbox[0] + self.bbox[2], self.bbox[1]))
        bbox_right = rendering.Line((self.bbox[0] + self.bbox[2], self.bbox[1]), (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]))
        bbox_up = rendering.Line((self.bbox[0], self.bbox[1] + self.bbox[3]), (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3]))
        bbox_left = rendering.Line((self.bbox[0], self.bbox[1]), (self.bbox[0], self.bbox[1] + self.bbox[3]))

        self.viewer_cam.add_geom(cam_up)
        self.viewer_cam.add_geom(cam_down)
        self.viewer_cam.add_geom(cam_left)
        self.viewer_cam.add_geom(cam_right)
        self.viewer_cam.add_geom(bbox_up)
        self.viewer_cam.add_geom(bbox_down)
        self.viewer_cam.add_geom(bbox_left)
        self.viewer_cam.add_geom(bbox_right)

        return self.viewer_env.render(return_rgb_array=mode == 'rgb_array'), self.viewer_cam.render(return_rgb_array=mode == 'rgb_array')

    def closer(self):
        if self.viewer_env:
            self.viewer_env.close()
            self.viewer_env = None






# register(
#     id="RobotInterception-v1",
#     entry_point="gym.envs.final:DiscreteRobotMovingEnv",
#     max_episode_steps=100,
#     reward_threshold=100,
# )



