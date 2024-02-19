import numpy as np
import random
import json

# Define constants
EMPTY = 0
OBSTACLE = 1
AGENT = 2
DESTINATION = 3
ACTIONS = ['up', 'down', 'left', 'right']

class QLearningAgent:
    def __init__(self, world_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.9):
        self.world_size = world_size
        self.q_table = np.zeros((world_size, world_size, len(ACTIONS)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state, mode='train'):
        if random.random() < self.exploration_rate:
            return random.choice(range(len(ACTIONS)))
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_value(self, state, action, reward, next_state):
        if next_state is None:
            # Handle case when next_state is None (agent hits obstacle)
            old_q_value = self.q_table[state[0], state[1], action]
            temporal_difference = reward - old_q_value
            self.q_table[state[0], state[1], action] += self.learning_rate * temporal_difference
        else:
            old_q_value = self.q_table[state[0], state[1], action]
            temporal_difference = reward + self.discount_factor * np.max(self.q_table[next_state[0], next_state[1]]) - old_q_value
            self.q_table[state[0], state[1], action] += self.learning_rate * temporal_difference

    def save_q_table(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.q_table.tolist(), f)

    def load_q_table(self, file_name):
        with open(file_name, 'r') as f:
            self.q_table = np.array(json.load(f))


class GridWorld:
    def __init__(self, world_size):
        self.world_size = world_size
        self.world = np.zeros((world_size, world_size))
        self.agent_pos = None
        self.destination_pos = (world_size - 1, world_size - 1)
        self.obstacles = [[1,1]]


        # Add obstacles randomly
        num_obstacles = random.randint(world_size // 2, world_size * 2)
        for _ in range(num_obstacles):
            row, col = random.randint(0, world_size - 1), random.randint(0, world_size - 1)
            if (row, col) != self.destination_pos:
                self.world[row, col] = OBSTACLE

        # Place agent randomly
        while True:
            row, col = random.randint(0, world_size - 1), random.randint(0, world_size - 1)
            if (row, col) != self.destination_pos and self.world[row, col] != OBSTACLE:
                self.agent_pos = (row, col)
                break

    def reset(self):
        self.agent_pos = (0, 0)

    def is_valid_move(self, pos):
        return 0 <= pos[0] < self.world_size and 0 <= pos[1] < self.world_size and self.world[pos[0], pos[1]] != OBSTACLE

    def move_agent(self, action):
        new_pos = self.agent_pos
        if action == 'up':
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 'down':
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 'left':
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 'right':
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)

        if 0 <= new_pos[0] < self.world_size and 0 <= new_pos[1] < self.world_size:
            # Move dynamic obstacles
            for i, obstacle in enumerate(self.obstacles):
                row, col = obstacle
                new_row, new_col = row, col
                while (new_row, new_col) == self.agent_pos or (new_row, new_col) in self.obstacles:
                    direction = random.choice(['up', 'down', 'left', 'right'])
                    if direction == 'up' and row > 0:
                        new_row = row - 1
                    elif direction == 'down' and row < self.world_size - 1:
                        new_row = row + 1
                    elif direction == 'left' and col > 0:
                        new_col = col - 1
                    elif direction == 'right' and col < self.world_size - 1:
                        new_col = col + 1
                self.obstacles[i] = (new_row, new_col)

            # Move agent
            if self.world[new_pos[0], new_pos[1]] == OBSTACLE:
                self.agent_pos = None  # Game over if agent hits obstacle
            else:
                self.agent_pos = new_pos

    def get_state(self):
        return self.agent_pos

    def get_reward(self):
        if self.agent_pos == self.destination_pos:
            return 100
        elif self.agent_pos == None:
            return -100  # Game over if agent hits obstacle
        else:
            return -1

    def print_world(self):
        for row in range(self.world_size):
            for col in range(self.world_size):
                if (row, col) == self.agent_pos:
                    print('A', end=' ')
                elif (row, col) == self.destination_pos:
                    print('D', end=' ')
                elif self.world[row, col] == OBSTACLE:
                    print('X', end=' ')
                else:
                    print('-', end=' ')
            print()


def main():
    # Define parameters
    world_size = 6
    num_episodes = 1000

    # Initialize grid world and agent
    world = GridWorld(world_size)
    agent = QLearningAgent(world_size)

    # User input
    mode = input("Choose mode (train/play): ")

    if mode == 'train':
        # Training loop
        for episode in range(num_episodes):
            world.reset()
            world.print_world()
            total_reward = 0
            while True:
                state = world.get_state()
                action = agent.choose_action(state, mode='train')
                world.move_agent(ACTIONS[action])
                world.print_world()
                print("\n")
                reward = world.get_reward()
                total_reward += reward
                next_state = world.get_state()
                agent.update_q_value(state, action, reward, next_state)
                if world.get_state() == world.destination_pos:
                    print("Reached destination!")
                    break
                elif world.get_state() == None:
                    print("Game over! Agent hit an obstacle.")
                    break

        # Save Q-table
        agent.save_q_table("q_table.json")

    elif mode == 'play':
        # Load Q-table
        agent.load_q_table("q_table.json")

        # Test loop
        world.reset()
        world.print_world()
        while True:
            state = world.get_state()
            action = agent.choose_action(state, mode='play')
            world.move_agent(ACTIONS[action])
            world.print_world()
            print()
            if world.get_state() == world.destination_pos:
                print("Reached destination!")
                break
            elif world.get_state() == None:
                print("Game over! Agent hit an obstacle.")
                break

    else:
        print("Invalid mode. Please choose 'train' or 'play'.")


if __name__ == "__main__":
    main()
