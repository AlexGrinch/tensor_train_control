import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


class WindyGridWorld:

    def __init__(
            self,
            grid_size=(11, 14),
            stochasticity=0.1,
            visual=False):
        """
        Simple WindyGridWorld environment from https://arxiv.org/abs/1710.10044

        Parameters
        ----------
        grid_size: tuple of two ints (W, H)
            size of the GridWorld, W should be odd and H = W + 3
        stochasticity: float from [0, 1]
            probability to select random action instead of intended
        visual: bool
            if True, the states will be images
        """
        self.w, self.h = grid_size
        self.stochasticity = stochasticity
        self.visual = visual
        self.x_wall = self.w // 2  # x position of the wll
        self.y_hole = self.h - 4  # y position of the hole in the wall
        self.reset()

    def clip_xy(self, x, y):
        """clip coordinates if they go beyond the grid"""
        x_ = np.clip(x, 0, self.w - 1)
        y_ = np.clip(y, 0, self.h - 1)
        return x_, y_

    def wind_shift(self, x, y):
        """apply wind shift to areas where wind is blowing"""
        if x == 1:
            return self.clip_xy(x, y + 1)
        elif x > 1 and x < self.x_wall:
            return self.clip_xy(x, y + 2)
        else:
            return x, y

    def move(self, a):
        """find valid coordinates of the agent after executing action"""

        x, y = self.pos
        x, y = self.wind_shift(x, y)

        if a == 0:
            x_, y_ = x + 1, y
        if a == 1:
            x_, y_ = x, y + 1
        if a == 2:
            x_, y_ = x - 1, y
        if a == 3:
            x_, y_ = x, y - 1

        # check if new position does not conflict with the wall
        if x_ == self.x_wall and y_ != self.y_hole:
            x_, y_ = x, y
        return self.clip_xy(x_, y_)

    def get_observation(self):
        if self.visual:
            obs = np.rot90(self.field)[:, :, None].copy()
        else:
            obs = self.pos
        return obs

    def reset(self):
        """reset the environment"""
        self.field = np.zeros((self.w, self.h))
        self.field[self.x_wall, :] = 1
        self.field[self.x_wall, self.y_hole] = 0
        self.field[0, 0] = 2
        self.pos = (0, 0)
        obs = self.get_observation()
        return obs

    def step(self, a):
        """make a step in the environment"""

        if np.random.rand() < self.stochasticity:
            a = np.random.randint(4)

        self.field[self.pos] = 0
        self.pos = self.move(a)
        self.field[self.pos] = 2

        reward, done = self.get_reward_done()
        next_obs = self.get_observation()
        return next_obs, reward, done

    def get_reward_done(self):
        """get reward and one indicator"""
        if self.pos == (self.w - 1, 0):
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return reward, done

    def play_with_policy(self, policy, max_iter=100, visualize=True):
        """play with given policy, returns return and number of time steps"""
        self.reset()
        for i in range(max_iter):
            a = np.argmax(policy[self.pos])
            next_obs, reward, done = self.step(a)
            # plot grid world state
            if visualize:
                img = np.rot90(1-self.field)
                plt.imshow(img, cmap="gray")
                display.clear_output(wait=True)
                display.display(plt.gcf())
                time.sleep(0.01)
            if done:
                break
        if visualize:
            display.clear_output(wait=True)
        return reward, i+1
