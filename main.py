import numpy as np
import os


class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state, training=True):
        print(state)
        if training and np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.num_actions)  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def save_q_table(self, file_path):
        np.save(file_path, self.q_table)

    def load_q_table(self, file_path):
        if os.path.exists(file_path):
            self.q_table = np.load(file_path)
        else:
            print("File not found. Starting with a new Q table.")

class Environment:
    def __init__(self, world):
        self.world = world
        self.num_rows = len(world)
        self.num_cols = len(world[0])
        self.num_states = self.num_rows * self.num_cols
        self.obstacle_value = 'x'
        self.agent_value = 'A'
        self.destination_value = 'END'

    def state_to_index(self, row, col):
        return row * self.num_cols + col

    def index_to_state(self, index):
        return divmod(index, self.num_cols)

    def is_valid_move(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.world[row][col] != self.obstacle_value

    def get_reward(self, row, col):
        if self.world[row][col] == self.destination_value:
            return 1
        else:
            return 0

    def display_world(self, agent_row, agent_col):
        world_copy = [row.copy() for row in self.world]
        world_copy[agent_row][agent_col] = self.agent_value
        for row in world_copy:
            print(" ".join(row))

def main():
    # Define the world
    world = [
        [' ', 'x', ' ', ' '],
        [' ', ' ', 'x', ' '],
        [' ', 'x', ' ', ' '],
        [' ', ' ', ' ', 'END']
    ]

    env = Environment(world)
    agent = QLearningAgent(env.num_states, 4)  # Assuming 4 actions: up, down, left, right

    # Option to load Q table values from a file
    load_q_table = input("Do you want to load Q table values from a file? (y/n): ").lower().strip()
    if load_q_table == 'y':
        file_path = "q_table.npy"
        agent.load_q_table(file_path)

    num_episodes = 100
    max_steps_per_episode = 100

    mode = input("Select mode (training/testing): ").lower().strip()

    for episode in range(num_episodes):
        state = env.state_to_index(0, 0)  # Start at top-left corner
        total_reward = 0

        for step in range(max_steps_per_episode):
            print(f"Episode {episode + 1}, Step {step + 1}")
            env.display_world(*env.index_to_state(state))

            if mode == 'training':
                action = agent.choose_action(state)
            else:
                action = agent.choose_action(state, training=False)

            # Decode action
            if action == 0:  # Up
                next_row, next_col = env.index_to_state(state)[0] - 1, env.index_to_state(state)[1]
            elif action == 1:  # Down
                next_row, next_col = env.index_to_state(state)[0] + 1, env.index_to_state(state)[1]
            elif action == 2:  # Left
                next_row, next_col = env.index_to_state(state)[0], env.index_to_state(state)[1] - 1
            else:  # Right
                next_row, next_col = env.index_to_state(state)[0], env.index_to_state(state)[1] + 1

            if env.is_valid_move(next_row, next_col):
                next_state = env.state_to_index(next_row, next_col)
                reward = env.get_reward(next_row, next_col)
                if mode == 'training':
                    agent.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward

                if world[next_row][next_col] == env.destination_value:
                    print(f"Episode {episode + 1} finished after {step + 1} steps with total reward {total_reward}")
                    break
            else:
                # Invalid move, stay in the same state
                next_state = state

    # Option to save Q table values to a file
    save_q_table = input("Do you want to save Q table values to a file? (y/n): ").lower().strip()
    if save_q_table == 'y':
        file_path = "q_table"
        agent.save_q_table(file_path)

if __name__ == "__main__":
    main()
