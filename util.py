import numpy as np
import math

testM = np.array([
    [" ", "X", " "],
    ["A", " ", " "],
    ["X", "X", " "],
    [" ", "X", " "]
])

testM2 = np.array([
        [" ", " ", "X", " "],
        [" ", " ", "X", " "],
        [" ", " ", " ", " "],
        [" ", " ", " ", "X"],
        [" ", " ", "X", " "],
        [" ", " ", " ", " "],
        [" ", " ", "X", "A"],
        [" ", " ", " ", "END"]
        ])
testM2Pos = [6, 3]

testPos = [1,0]


'''
    env: The world enviornment: 2D Array
    agent_pos: The position of the agent in the world : [row, col]

    RETURNS
    [9 Elements] :
        1 == Obstacle
        0 == Empty Space
        -1 == Out of Bound
        10 == Goal
        5 == Agent
'''
pos = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0,0], [0,1], [1,-1], [1, 0], [1,1]]
visionPos = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
def env_to_vision(env, agent_pos, isTuple=False):
    vision_list = [0,0,0,0,0,0,0,0,0]
    direction = [True,True,True,True]
    num_col = len(env[0])

    for i in range(len(pos)):
        if is_outside_env(env, agent_pos[0] + pos[i][0], agent_pos[1] + pos[i][1]):
            #vision_list[state_to_index(visionPos[i][0], visionPos[i][1], num_col)] = -1
            vision_list[i] = -1
        elif env[agent_pos[0] + pos[i][0]][agent_pos[1] + pos[i][1]] == "X":
            #print(agent_pos[0] + pos[i][0], agent_pos[1] + pos[i][1])
            vision_list[i] = 1
        elif env[agent_pos[0] + pos[i][0]][agent_pos[1] + pos[i][1]] == "END":
            vision_list[i] = 10
        elif env[agent_pos[0] + pos[i][0]][agent_pos[1] + pos[i][1]] == "A":
            vision_list[i] = 5

            #vision_list[state_to_index(agent_pos[0] + pos[i][0], agent_pos[1] + pos[i][1] + 1, num_col)] = 1
    if isTuple:
        return tuple(vision_list)
    #print(vision_list)
    return vision_list
 


def is_outside_env(env, row, col):
    """
    Check if the given row and column are outside the boundaries of the environment.

    Parameters:
        env (2D array): The 2D array representing the environment.
        row (int): The row index to check.
        col (int): The column index to check.

    Returns:
        bool: True if the coordinate is outside the boundaries, False otherwise.
    """
    num_rows, num_cols = env.shape
    return row < 0 or row >= num_rows or col < 0 or col >= num_cols




def state_to_index(row, col, num_cols):
    return row * num_cols + col


#env_to_vision(testM2, testM2Pos)

def encouragement(agent_pos, old_agent_pos, destination, positive_reward=0, negative_reward=-1):
    new_dst = math.sqrt((destination[0] - agent_pos[0])**2 + (destination[1] - agent_pos[1])**2)
    old_dst = math.sqrt((destination[0] - old_agent_pos[0])**2 + (destination[1] - old_agent_pos[1])**2)

    if new_dst < old_dst:
        return positive_reward
    else:
        return negative_reward

