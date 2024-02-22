import numpy as np
import random
from obstacles import On_Off_Obstacle


class GridWorld:
    def __init__(self, width=12, length=12, obstacle_chance = 0.4):
        self.width = width
        self.length = length
        self.ob_chance = obstacle_chance
        self.dynamic_obstacles = []
        self.grid = np.full((self.width, self.length), '~')

    def generate_grid_world(self):
        # Place the agent at the top-left corner
        self.grid[0, 0] = 'A'
        
        # Place the destination at the bottom-right corner
        self.grid[self.width-1, self.length-1] = 'END'
        
        # Place an obstacle at a random position
        for i in range(10):
            obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)
            if (random.randint(0, 9) < self.ob_chance*10):
                # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner
                while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == self.width-1 and obstacle_y == self.length-1):
                    obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)

                self.grid[obstacle_x, obstacle_y] = 'X'

    def add_on_off_obstacles(self):
        # Place an obstacle at a random position
        for i in range(10):
            obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)
            if (random.randint(0, 9) < 9):
                # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner
                while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == self.width-1 and obstacle_y == self.length-1):
                    obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)

                temp = On_Off_Obstacle(self.grid, obstacle_x, obstacle_y, random.randint(1, 5))
                temp.plot()
                self.dynamic_obstacles.append(temp)

    def tick(self):
        # Function for ticks
        for obj in self.dynamic_obstacles:
            obj.tick()


# Generate the grid world
grid_world = GridWorld()
grid_world.generate_grid_world()
grid_world.add_on_off_obstacles()

for i in range(5):
    print(grid_world.grid)
    print("-----------------------------------")
    grid_world.tick()
