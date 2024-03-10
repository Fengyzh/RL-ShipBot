import numpy as np
import random
from obstacles import On_Off_Obstacle
from obstacles import Moving_Obstacle

class GridWorld:
    def __init__(self, width=12, length=12, obstacle_chance = 0.4):
        self.width = width
        self.length = length
        self.ob_chance = obstacle_chance
        self.dynamic_obstacles = []
        self.dynamic_obs  = {}
        self.goal = (self.width-1, self.length-1)
        self.grid = np.full((self.width, self.length), '~')

    def static_map_test(self, number):
        if number == 1:
            self.grid = np.array([        # The world
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "E",]
            ])
        if number == 2:
            self.grid = np.array([        # The world
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "X", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "X", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "X", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "X",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "X", "~", "E",]
            ])

        if number == 3:
            self.grid = np.array([        # The world
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "X", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "X", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "X", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "X",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~", "~",],
                ["~", "~", "~", "~", "~", "~", "~", "~", "~", "X", "~", "E",]
            ])

            coords_x = [5,3,10,11,10]
            coords_y = [5,9,5,0,8]
            # Place an obstacle at a random position
            for i in range(5):
                obstacle_x, obstacle_y = coords_x[i], coords_y[i]
                # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner or conflicting with another object's spawn position
                while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == self.width-1 and obstacle_y == self.length-1) or self.conflict_of_position_check(obstacle_x, obstacle_y):
                    obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)

                temp = On_Off_Obstacle(self.grid, obstacle_x, obstacle_y, random.randint(3, 5))
                temp.plot()
                self.dynamic_obs[(temp.get_obs_pos)] = temp

    def generate_grid_world(self, onOffObstacles=[], movingObstacles=[]):
        # Place the destination at the bottom-right corner
        self.grid[self.goal[0], self.goal[1]] = 'E'
        
        # Place an obstacle at a random position
        for i in range(10):
            obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)
            if (random.randint(0, 9) < self.ob_chance*10):
                # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner
                while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == self.width-1 and obstacle_y == self.length-1):
                    obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)

                self.grid[obstacle_x, obstacle_y] = 'X'

        if (onOffObstacles):
            self.add_on_off_obstacles()

        if (movingObstacles):
            self.add_moving_obstacles()

    def add_on_off_obstacles(self):
        # Place an obstacle at a random position
        for i in range(10):
            obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)
            if (random.randint(0, 9) < 9):
                # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner or conflicting with another object's spawn position
                while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == self.width-1 and obstacle_y == self.length-1) or self.conflict_of_position_check(obstacle_x, obstacle_y):
                    obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)

                temp = On_Off_Obstacle(self.grid, obstacle_x, obstacle_y, random.randint(3, 5))
                temp.plot()
                self.dynamic_obs[(temp.get_obs_pos)] = temp

    def add_moving_obstacles(self):
        # Place an obstacle at a random position
        for i in range(8):
            obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)
            if (random.randint(0, 9) < 9):
                # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner or conflicting with another object's spawn position
                while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == self.width-1 and obstacle_y == self.length-1) or self.conflict_of_position_check(obstacle_x, obstacle_y):
                    obstacle_x, obstacle_y = random.randint(0, self.width-1), random.randint(0, self.length-1)

                temp = Moving_Obstacle(self.grid, obstacle_x, obstacle_y, random.choice(['up', 'left', 'right', 'down']))
                temp.plot()
                self.dynamic_obs[temp.get_obs_pos] = temp

    def set_start_pos(self, x, y):
        if self.conflict_of_position_check(x,y):
            self.dynamic_obs.pop((y,x))
            self.grid[x,y] = "A"
        else:
            self.grid[x,y] = "A"

    def conflict_of_position_check(self, x, y):
        if self.dynamic_obs.get((y,x)) is None:
            return False
        else:
            return True

    def tick(self):
        # Function for ticks
        for key in self.dynamic_obs.keys():
            self.dynamic_obs[key].tick()



if __name__ == "__main__":
    # Generate the grid world
    grid_world = GridWorld()
    grid_world.generate_grid_world(True, True)
    grid_world.set_start_pos(2,2)

    for i in range(5):
        print(grid_world.grid)
        print("-----------------------------------")
        grid_world.tick()
