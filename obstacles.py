import numpy as np


'''
times: how many times it should turn on and off
'''
class On_Off_Obstacle:
    def __init__(self, env, x, y, end=1, times=1):
        self.time = times
        self.counter = 0
        self.env = env
        self.x = x
        self.y = y
        self.end = end
    
    def tick(self):
        self.counter += 1
        self.plot()

    def plot(self):
        if self.counter < self.end:
            self.env[self.x, self.y] = 'X'
        else:
            self.env[self.x, self.y] = ' '
