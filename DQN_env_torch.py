import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from grid import GridWorld
from util import encouragement
from util import env_to_vision
from metric import Metrics

DEFAULT_AGNET_POS = (6, 6)
DEFAULT_DES_POS = (11,11)
MAX_STEP = 100

gridMap = GridWorld()
gridMap.generate_grid_world()
gridMap.set_start_pos(DEFAULT_AGNET_POS[0], DEFAULT_AGNET_POS[1])
envMap = gridMap.grid

sm = np.array([
    [" ", " ", "X", " "],
    [" ", " ", "X", " "],
    [" ", " ", " ", " "],
    [" ", " ", " ", "X"],
    [" ", " ", " ", "X"],
    [" ", " ", " ", " "],
    [" ", " ", "X", " "],
    [" ", " ", " ", "END"]
])

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
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if self.swap_count == SWAP_COUNT:
            self.model.load_state_dict(self.target_model.state_dict())
            self.swap_count = 0
        else:
            self.swap_count += 1

        minibatch = random.sample(self.memory, batch_size)
        X, y = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model(torch.FloatTensor(state))
            if done:
                target[0][action] = reward
            else:
                next_state_value = self.target_model(torch.FloatTensor(next_state)).max(1)[0].detach()
                target[0][action] = reward + self.gamma * next_state_value
            X.append(state)
            y.append(target)

        X = torch.FloatTensor(X)  # Convert X_batch to Tensor
        y = torch.stack(y)
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        output = self.model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()


    def train(self, state, action, reward, next_state):
        target = self.model(torch.FloatTensor(state))
        t = self.model(torch.FloatTensor(next_state))[0]
        target[0][action] = reward + self.gamma * torch.max(t)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        output = self.model(torch.FloatTensor(state))
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

class Environment:
    def __init__(self):
        self.grid = envMap
        self.vision_grid = self.grid
        self.agent_position = DEFAULT_AGNET_POS
        self.destination = DEFAULT_DES_POS
        self.state_size = 9
        self.grid_mapping = {' ': 0, 'X': 1, 'END': 2, 'A': 3, '~': 0, 'E':2}
        self.stepCount = 0
        self.temp_agent_pos = DEFAULT_AGNET_POS

    def preprocess_state(self):
        self.vision_grid = self.state_to_vision()
        return [self.vision_grid]

    def reset(self):
        self.agent_position = DEFAULT_AGNET_POS
        self.temp_agent_pos = DEFAULT_AGNET_POS
        self.stepCount = 0
        self.grid = envMap.copy()

    def step(self, action):
        if action == 0:
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:
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
            reward = -100
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
            return 100
        elif self.grid[self.agent_position] == "X":
            return -100
        else:
            return encouragement(self.agent_position, self.temp_agent_pos, self.destination, False, self.stepCount, 1, -1)

    def _is_done(self):
        return self.grid[self.agent_position] == "E" or self.grid[self.agent_position] == "X"

    def state_to_vision(self):
        return env_to_vision(self.grid, self.agent_position, False)

def train():
    EPISODES = 200
    BATCH_SIZE = 16

    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()


    try:
        agent.load("trained_model_env.pth")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")

    for episode in range(EPISODES):
        state = env.preprocess_state()
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

        m.recordIteration(total_reward, True if env.agent_position == env.destination else False, step)
        print(f"Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward}")
        env.reset()

    agent.save("trained_model_env.pth")
    m.printMetrics()


def play():
    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    try:
        agent.load("trained_model_env.pth")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")

    for episode in range(2):
        print("-------------- Start Iter ------------------")
        print(env.grid)

        state = env.preprocess_state()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.pick_action(state)
            reward, done = env.step(action)
            total_reward += reward
            state = env.preprocess_state()
            print("\n")
            print(env.grid)
            step += 1

        print(f"Total Reward: {total_reward}")
        env.reset()

usr = input("Train/Play (Enter: t or p): ")
if usr == "t":
    train()
elif usr == "p":
    play()
