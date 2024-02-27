import numpy as np


# Note:
# The X and Y is fliped for obstcles
# ie: X +- 1 is going down and up, Y +- 1 is going right and left

'''
times: how many times it should turn on and off
x, y: x y cords
end: how many tick to end
'''
class On_Off_Obstacle:
    def __init__(self, env, y, x, end=1, times=1):
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
            self.env[self.x, self.y] = '~'
    
    def get_obs_pos(self):
        return (self.y, self.x)



class Moving_Obstacle:
    def __init__(self, env, y, x, dir='up', end=-1):
        self.dir = dir
        self.counter = 0
        self.env = env
        self.x = x
        self.y = y
        self.end = end
        self.num_rows, self.num_cols = env.shape
        self.stop = False
    
    def tick(self):
        self.counter += 1
        self.plot()

    def plot(self):
        # Init plot
        if self.counter < 1 or self.stop:
            self.env[self.x,self.y] = 'X'
            return
        
        temp = [self.x, self.y]
        if self.dir == 'left':
            self.y -= 1
        if self.dir == 'right':
            self.y += 1
        if self.dir == 'up':
            self.x -= 1
        if self.dir == 'down':
            self.x += 1
        

        if self.end < 0:
            if 0 <= self.x < len(self.env) and 0 <= self.y < len(self.env[0]) and self.env[self.x, self.y] != 'X':
                self.env[self.x, self.y] = 'X'
                self.env[temp[0], temp[1]] = '~'
            else:
                self.stop = True
            
        elif self.counter < self.end:
            self.env[self.x, self.y] = 'X'
        else:
            self.stop = True
        

    def get_obs_pos(self):
        return (self.y, self.x)