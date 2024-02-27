import numpy as np
import random
import json
from util import env_to_vision
from grid import GridWorld

# Define constants
EMPTY = 0
OBSTACLE = 1
AGENT = 2
DESTINATION = 3
ACTIONS = ['up', 'down', 'left', 'right']


# Refact out world_size
# State should be 0-9 representing the surrrounding spaces
class QLearningAgent:
    def __init__(self, world_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.world_size = world_size
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    # Refact out mode
    def choose_action(self, state):
        #print("Q: ", self.q_table)
        print("state: ", state)
        if (state not in self.q_table):
            self.q_table[state] = [0,0,0,0]
            return random.choice(range(len(ACTIONS)))
        elif random.random() < self.exploration_rate:
            return random.choice(range(len(ACTIONS)))
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        print(state, action)
        print(self.q_table)
        if next_state is None:
            # Handle case when next_state is None (agent hits obstacle)
            old_q_value = self.q_table[state][action]
            temporal_difference = reward - old_q_value
            self.q_table[state][action] += self.learning_rate * temporal_difference
        else:
            old_q_value = self.q_table[state][action]
            temporal_difference = reward + self.discount_factor * np.max(self.q_table[state]) - old_q_value
            self.q_table[state][action] += self.learning_rate * temporal_difference

    def save_q_table(self, file_name):
        q_table_json = {str(key): value for key, value in self.q_table.items()}

        # Save the Q-table to a JSON file
        with open('q_table.json', 'w') as json_file:
            json.dump(q_table_json, json_file)

    def load_q_table(self, file_name):
        with open('q_table.json', 'r') as json_file:
            q_table_json = json.load(json_file)

        # Convert string keys back to tuples
        self.q_table = {eval(key): value for key, value in q_table_json.items()}


class Environment:
    def __init__(self, world_size):
        self.world_size = world_size
        self.agent_pos = (0,0)
        self.destination_pos = (world_size - 1, world_size - 1)
        self.done = False

        # Builds Map
        self.gridWorld = GridWorld(self.world_size, self.world_size)
        self.gridWorld.generate_grid_world(True, True)
        self.world = self.gridWorld.grid

    def tick(self):
        self.gridWorld.tick()
        self.world = self.gridWorld.grid

    def reset(self):
        self.agent_pos = (0, 0)

    def is_valid_move(self, pos):
        return 0 <= pos[0] < len(self.world) and 0 <= pos[1] < len(self.world[0]) and self.world[pos[0], pos[1]] != 'X'

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

        self.world[self.agent_pos] = '~'

        if 0 <= new_pos[0] < len(self.world) and 0 <= new_pos[1] < len(self.world[0]):
            # Move agent
            if self.world[new_pos[0], new_pos[1]] == 'X' or self.world[new_pos[0], new_pos[1]] == 'END':
                self.agent_pos = new_pos  # Game over if agent hits obstacle
                self.done = True
            else:
                self.agent_pos = new_pos
                self.world[self.agent_pos] = 'A'
    
    def get_pos(self):
        return self.agent_pos
    
    def get_state(self):
        return env_to_vision(self.world, self.agent_pos, True)

    def get_reward(self):
        if self.world[self.agent_pos[0], self.agent_pos[1]] == 'END':
            return 100
        elif self.world[self.agent_pos[0], self.agent_pos[1]] == 'X':
            return -100  # Game over if agent hits obstacle
        else:
            return -1


    def print_world(self):
        rendered_grid = np.copy(self.world)
        rendered_grid[self.agent_pos] = "A"  # Render agent position
        print(rendered_grid)


def main():
    # Define parameters
    world_size = 12
    num_episodes = 1

    # Initialize grid environment and agent
    environment = Environment(world_size)
    agent = QLearningAgent(world_size)

    # User input
    mode = input("Choose mode (t/p): ")

    if mode == 't':
        # Training loop
        for episode in range(num_episodes):
            world.reset()
            world.print_world()
            print("\n")

            total_reward = 0
            while not world.done:
                state = world.get_state()
                print("train state: ", state)
                action = agent.choose_action(state)
                print(action)
                world.move_agent(ACTIONS[action])

                # Grid.tick
                world.tick()

                print("PPPPPPPPOS: ", world.agent_pos)
                print("\n")
                world.print_world()
                print("\n")
                reward = world.get_reward()
                total_reward += reward
                next_state = world.get_state()
                agent.update_q_value(state, action, reward, next_state)
                if world.world[world.get_pos()] == "END":
                    print("Reached destination!")
                    break
                elif world.world[world.get_pos()] == 'X' :
                    print("Game over! Agent hit an obstacle.")
                    break
            print(f"Total Reward: {total_reward}")

        # Save Q-table
        agent.save_q_table("q_tableq.json")

    elif mode == 'p':
        # Load Q-table
        agent.load_q_table("q_tableq.json")

        # Test loop
        environment.reset()

        while True:
            state = environment.get_state()
            action = agent.choose_action(state)
            environment.move_agent(ACTIONS[action])

            # Grid.tick
            environment.tick()

            environment.print_world()
            print()
            if environment.world[environment.get_pos()] == "END":
                print("Reached destination!")
                break
            elif environment.world[environment.get_pos()] == 'X' :
                print("Game over! Agent hit an obstacle.")
                break

    else:
        print("Invalid mode. Please choose 'train' or 'play'.")


if __name__ == "__main__":
    main()
