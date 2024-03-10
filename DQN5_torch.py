import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from obstacles import On_Off_Obstacle, Moving_Obstacle
from metric import Metrics
from grid import GridWorld
from util import encouragement
import matplotlib.pyplot as plt


DEFAULT_AGENT_POS = (0, 0)
DEFAULT_DES_POS = (7, 7)
MAX_STEP = 1000


allScore = []


gridMap = GridWorld(8,8)
gridMap.generate_grid_world(False, False)
#gridMap.static_map_test(2)
gridMap.set_start_pos(DEFAULT_AGENT_POS[0], DEFAULT_AGENT_POS[1])
envMap = gridMap.grid

""" envMap = np.array(
[['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', 'X', '~', '~', '~', 'X', '~'],
 ['~', 'X', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', 'X', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', 'A', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~'],
 ['~', '~', '~', '~', '~', '~', '~', '~', '~', '~', '~', 'E']]) """




SWAP_COUNT = 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 0.6
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.swap_count = 0

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def pick_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        X = []
        y = []
        if self.swap_count == SWAP_COUNT:
            self.model.load_state_dict(self.target_model.state_dict())
            self.swap_count = 0
        else:
            self.swap_count += 1

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            target = self.target_model(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model(torch.FloatTensor(next_state))[0]
                target[0][action] = reward + self.gamma * torch.max(t)
            X.append(state)
            y.append(target)
        X = torch.cat(X)
        y = torch.cat(y)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.target_model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        outputs = self.target_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))


class Environment:
    def __init__(self):
        self.grid = envMap.copy()
        self.agent_position = DEFAULT_AGENT_POS
        self.destination = DEFAULT_DES_POS
        self.state_size = np.prod(self.grid.shape)
        self.grid_mapping = {' ': 0, 'X': 1, 'END': 2, 'A': 3, '~': 0, 'E': 2}
        self.stepCount = 0
        self.temp_agent_pos = DEFAULT_AGENT_POS

    def tick(self):
        gridMap.tick()
        self.grid = gridMap.grid

    def preprocess_state(self):
        processed_grid = np.vectorize(self.grid_mapping.get)(self.grid)
        return processed_grid.flatten().reshape(1, -1)

    def reset(self):
        self.agent_position = DEFAULT_AGENT_POS
        self.temp_agent_pos = DEFAULT_AGENT_POS
        self.stepCount = 0
        self.grid = envMap.copy()

    def step(self, action):
        if action == 0:  # Move up
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Move down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:  # Move left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:  # Move right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)

        self.temp_agent_pos = self.agent_position
        if self._is_valid_move(new_position):
            self.grid[self.agent_position] = '~'
            self.agent_position = new_position
            if (self.grid[self.agent_position] != 'E' and self.grid[self.agent_position] != 'X'):
                self.grid[self.agent_position] = 'A'

        reward = self._get_reward()
        done = self._is_done()

        if self.stepCount >= MAX_STEP:
            done = True

        self.stepCount += 1

        return reward, done

    def _is_valid_move(self, position):
        row, col = position
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            return True
        return False

    def _get_reward(self):
        if self.grid[self.agent_position] == "E":
            return 1000  # Reward for reaching the destination
        elif self.grid[self.agent_position] == "X":
            return -500  # Penalty for hitting an obstacle
        else:
            return encouragement(self.agent_position, self.temp_agent_pos, self.destination, False, self.stepCount, 0, -1)

    def _is_done(self):
        return self.grid[self.agent_position] == "E" or self.grid[self.agent_position] == "X"

    def render(self):
        rendered_grid = np.copy(self.grid)
        rendered_grid[self.agent_position] = "A"  # Render agent position
        print(rendered_grid)


def train():
    EPISODES = 100
    BATCH_SIZE = 32

    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()

    try:
        agent.load("trained_model.pth")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")

    for episode in range(EPISODES):
        state = env.preprocess_state()  # Initial state
        total_reward = 0
        done = False
        step = 0

        print(env.grid)

        while not done:
            action = agent.pick_action(state)
            reward, done = env.step(action)
            total_reward += reward
            next_state = env.preprocess_state()
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
            state = next_state
            step += 1

        m.recordIteration(total_reward, True if total_reward > 0 else False, step)
        print(f"Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward}")
        env.reset()

    agent.save("trained_model.pth")
    m.printMetrics()


def play():
    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()

    #print(env.grid)


    agent.load("trained_model.pth")
    print("Loaded model from disk")

    print("No pre-trained model found, starting training from scratch.")

    for episode in range(200):
        print("-------------- Start Iter ------------------")

        #print(env.grid)

        state = env.preprocess_state()  # Initial state
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.pick_action(state)
            reward, done = env.step(action)
            total_reward += reward
            state = env.preprocess_state()
            #print("\n")
            #print(env.grid)
            env.tick()
            step += 1
        m.recordIteration(total_reward, True if env.agent_position == env.destination else False, step)
        print(f"Total Reward: {total_reward}")
        env.reset()
    m.printMetrics()
    allScore.append(m.success)
    print(env.grid)


usr = input("Train/Play (Enter: t or p): ")
if usr == "t":
    for i in range(1):
        train()
elif usr == "p":
    for i in range(10):
        play()
# Generating x-values (arbitrary)
    x_values = range(len(allScore))

    # Create scatter plot
    plt.scatter(x_values, allScore)
    plt.ylim(0, 200)

    # Add labels and title
    plt.xlabel('Runs')
    plt.ylabel('Scores')
    plt.title('Scores Plot')

    # Show plot
    plt.show()