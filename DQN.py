import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# BEFORE YOU START
# Make sure your file_path_str is point to the directory this project's DQNmodel.pth
# For saving and loading the model



# Define the environment
env = np.array([
    ['X', 'X', 'X', 'X', 'X', 'X'],
    ['X', ' ', ' ', ' ', ' ', 'X'],
    ['X', ' ', 'X', 'X', ' ', 'X'],
    ['X', ' ', ' ', 'X', ' ', 'X'],
    ['X', 'X', ' ', ' ', ' ', 'X'],
    ['X', 'X', 'X', 'X', ' ', 'END']
])

num_rows, num_cols = env.shape

# Define Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define action mapping
actions = {
    0: 'up',
    1: 'down',
    2: 'left',
    3: 'right'
}

# Define DQN agent
class DQNAgent:
    def __init__(self, model, epsilon=0.9):
        self.model = model
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(output_size)
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()

# Define state representation function
def get_state(env, pos):
    state = []
    for i in range(pos[0] - 1, pos[0] + 2):
        for j in range(pos[1] - 1, pos[1] + 2):
            if i < 0 or i >= env.shape[0] or j < 0 or j >= env.shape[1]:
                state.append(1)  # Boundary
            elif env[i][j] == 'X':
                state.append(2)  # Obstacle
            elif env[i][j] == ' ':
                state.append(0)  # Empty space
            elif env[i][j] == 'END':
                state.append(3)  # Destination
            elif env[i][j] == 'A':
                state.append(0)  # Agent position
    return state

# Define function to update environment based on action
def update_env(env, pos, action):
    new_pos = pos[:]
    if action == 0:  # Up
        new_pos[0] -= 1
    elif action == 1:  # Down
        new_pos[0] += 1
    elif action == 2:  # Left
        new_pos[1] -= 1
    elif action == 3:  # Right
        new_pos[1] += 1
    
    if new_pos[0] < num_rows and new_pos[1] < num_cols and env[new_pos[0]][new_pos[1]] != 'X' :
        env[pos[0]][pos[1]] = ' '
        env[new_pos[0]][new_pos[1]] = 'A'
        pos[:] = new_pos
        return False
    return True

# Training function
def train_agent(agent, model, optimizer, num_episodes=1000, gamma=0.2, load_model_path=None):
    if load_model_path:
        load_model(model, load_model_path)
        
    for episode in range(num_episodes):
        env_copy = env.copy()
        agent_pos = [5, 4]  # Starting position
        done = False
        total_reward = 0
        
        while not done:
            state = get_state(env_copy, agent_pos)
            action = agent.select_action(state)
            reward = -1  # Default reward for each step
            
            # Update environment based on action
            done = update_env(env_copy, agent_pos, action)
            
            # Calculate reward
            if done and env_copy[agent_pos[0]][agent_pos[1]] == 'END':
                reward = 100  # Reached the destination
            elif done:
                reward = -100  # Collided with an obstacle
            
            total_reward += reward
            
            # Update Q-values
            next_state = get_state(env_copy, agent_pos)
            q_values = model(torch.tensor(state, dtype=torch.float32))
            next_q_values = model(torch.tensor(next_state, dtype=torch.float32))
            max_next_q_value = torch.max(next_q_values).item()
            target_q_value = reward + gamma * max_next_q_value # Bellman Algo
            q_values[action] = target_q_value
            
            # Loss calculation and optimization
            # Using the Q_value from the bellman algo to calulate loss and optimize
            loss = nn.MSELoss()(q_values, model(torch.tensor(state, dtype=torch.float32)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Play function
def play_agent(agent, model):
    env_copy = env.copy()
    agent_pos = [5, 4]  # Starting position
    done = False
    total_reward = 0
    reward = 0
    
    while not done:
        state = get_state(env_copy, agent_pos)
        action = agent.select_action(state)
        print(f"Action: {actions[action]}")
        
        # Update environment based on action
        done = update_env(env_copy, agent_pos, action)
        
        # Display environment
        for row in env_copy:
            print(' '.join(row))
        
        # Calculate reward
        if done and env_copy[agent_pos[0]][agent_pos[1]] == 'END':
            reward = 100  # Reached the destination
        elif done:
            reward = -100  # Collided with an obstacle
        
        total_reward += reward
        
    print(f"Total Reward: {total_reward}")

# Save model function
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# Load model function
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")

# Main function
if __name__ == "__main__":
    input_size = 9  # Number of features (surrounding cells + current cell)
    output_size = 4  # Number of possible actions (up, down, left, right)
    file_path_str = r"D:\vs_code_projects\CS462_RL\DQNmodel.pth"
    model_filepath = Path(file_path_str)    
    model = QNetwork(input_size, output_size) # Initialize Q-network
    
    # Initialize agent
    agent = DQNAgent(model)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.11)
    
    mode = input("Choose mode (train/play): ")
    
    if mode == "train":
        num_episodes = int(input("Enter number of episodes: "))
        load_model_path = input("Load model for train? (yes for yes, empty for no): ")
        if not load_model_path:
            load_model_path = "" 
        elif load_model_path == "yes":
            load_model_path = model_filepath
        train_agent(agent, model, optimizer, num_episodes=num_episodes, load_model_path=load_model_path or None)
        save = input("Save model? (yes/no): ")
        if save == "yes":
            save_model(model, model_filepath)
    elif mode == "play":
        load = input("Load model? (yes/no): ")
        if load == "yes":
            load_model(model, model_filepath)
        play_agent(agent, model)
    else:
        print("Invalid mode. Please choose 'train' or 'play'.")
