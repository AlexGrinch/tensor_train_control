import numpy as np
from .wgw_env import WindyGridWorld


class EvilWindyGridWorld(WindyGridWorld):

    def __init__(
            self,
            grid_size=(7, 10),
            stochasticity=0.1,
            visual=False):
        """
        Evil WindyGridWorld environment which has an additional hole in
        the wall and two traps with a reward of -1. Optimal policy now
        depends on stochasticity: if stoc < 0.04, the optimal policy is
        going through the bottom hole; otherwise, the optimal policy is
        going through the top hole.

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
        self.y_hole2 = self.h - 7  # y position of the second hole
        self.reset()

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
        if x_ == self.x_wall and y_ != self.y_hole and y_ != self.y_hole2:
            x_, y_ = x, y
        return self.clip_xy(x_, y_)

    def reset(self):
        """reset the environment"""
        self.field = np.zeros((self.w, self.h))
        self.field[self.x_wall, :] = 1
        self.field[self.x_wall, self.y_hole] = 0
        self.field[self.x_wall, self.y_hole2] = 0
        self.field[self.x_wall + 1, self.y_hole2 + 1] = -1
        self.field[self.x_wall + 1, self.y_hole2 - 1] = -1
        self.field[0, 0] = 2
        self.pos = (0, 0)
        obs = self.get_observation()
        return obs

    def get_reward_done(self):
        """get reward and one indicator"""
        reward, done = super(EvilWindyGridWorld, self).get_reward_done()
        if (self.pos == (self.x_wall + 1, self.y_hole2 + 1) or
                self.pos == (self.x_wall + 1, self.y_hole2 - 1)):
            reward = -1
            done = True
        return reward, done

    def set_pos(self, pos):
        """put the agent into particular position in the field"""
        self.reset()
        self.field[0, 0] = 0
        self.pos = pos
        self.field[pos[0], pos[1]] = 2
        obs = self.get_observation()
        valid = True
        if (pos == (self.w-1, 0) or \
                pos == (self.x_wall+1, self.y_hole2+1) or \
                pos == (self.x_wall+1, self.y_hole2-1)):
            valid = False
        if (pos[0] == self.x_wall and pos[1] != self.y_hole \
                and pos[1] != self.y_hole2):
            valid = False
        return valid, obs
