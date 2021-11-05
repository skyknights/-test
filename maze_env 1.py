"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 50   # pixels
MAZE_H = 8  # grid height
MAZE_W = 8 # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

         #hell1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        
        # hell2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
             hell2_center[0] - 15, hell2_center[1] - 15,
             hell2_center[0] + 15, hell2_center[1] + 15,
             fill='black')
       


        # hell3
        hell3_center = origin + np.array([UNIT, UNIT * 4])
        self.hell3 = self.canvas.create_rectangle(
             hell3_center[0] - 15, hell3_center[1] - 15,
             hell3_center[0] + 15, hell3_center[1] + 15,
             fill='black')

        # hell4
        hell4_center = origin + np.array([UNIT*3, UNIT * 6])
        self.hell4 = self.canvas.create_rectangle(
             hell4_center[0] - 15, hell4_center[1] - 15,
             hell4_center[0] + 15, hell4_center[1] + 15,
             fill='black')
        
        # hell5
        hell5_center = origin + np.array([UNIT*7, UNIT * 7])
        self.hell5 = self.canvas.create_rectangle(
             hell5_center[0] - 15, hell5_center[1] - 15,
             hell5_center[0] + 15, hell5_center[1] + 15,
             fill='black')

        
        # hell6
        hell6_center = origin + np.array([UNIT*4, UNIT * 4])
        self.hell6 = self.canvas.create_rectangle(
             hell6_center[0] - 15, hell6_center[1] - 15,
             hell6_center[0] + 15, hell6_center[1] + 15,
             fill='black')

        
        # hell7
        hell7_center = origin + np.array([UNIT*5, UNIT * 2])
        self.hell7 = self.canvas.create_rectangle(
             hell7_center[0] - 15, hell7_center[1] - 15,
             hell7_center[0] + 15, hell7_center[1] + 15,
             fill='black')
        
        # hell8
        hell8_center = origin + np.array([UNIT*7, UNIT * 4])
        self.hell8 = self.canvas.create_rectangle(
             hell8_center[0] - 15, hell8_center[1] - 15,
             hell8_center[0] + 15, hell8_center[1] + 15,
             fill='black')

        # create oval
        oval_center = origin + UNIT * 6
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 2
            done = True
        elif next_coords in [self.canvas.coords(self.hell1),self.canvas.coords(self.hell2),
self.canvas.coords(self.hell3),self.canvas.coords(self.hell4),
self.canvas.coords(self.hell5),self.canvas.coords(self.hell6),
self.canvas.coords(self.hell6),self.canvas.coords(self.hell8)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


