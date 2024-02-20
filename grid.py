import numpy as np
import random
from obstacles import On_Off_Obstacle

def generate_grid_world():
    # Define the size of the grid
    grid_size = 5
    
    # Initialize the grid with spaces
    grid = np.full((grid_size, grid_size), ' ')
    
    # Place the agent at the top-left corner
    grid[0, 0] = 'A'
    
    # Place the destination at the bottom-right corner
    grid[grid_size-1, grid_size-1] = 'END'
    
    # Place an obstacle at a random position
    for i in range(10):
        obstacle_x, obstacle_y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
    # Make sure the obstacle is not placed at the top-left corner or the bottom-right corner

        while (obstacle_x == 0 and obstacle_y == 0) or (obstacle_x == grid_size-1 and obstacle_y == grid_size-1):
            obstacle_x, obstacle_y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        grid[obstacle_x, obstacle_y] = 'X'
    
    return grid

# Generate the grid world
grid_world = generate_grid_world()

# Print the grid world
#print(grid_world)




o = On_Off_Obstacle(grid_world, 2, 3)
o.plot()

for i in range(2):
    print(grid_world)
    o.tick()
