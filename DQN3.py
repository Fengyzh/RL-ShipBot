import tensorflow as tf
from collections import deque
import numpy as np
import random

DEFAULT_AGNET_POS = (7,1)
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

# Create Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def pick_action(self, state):
        print(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)

    def train(self, state, action, reward, next_state):
        target = self.model.predict(state)
        t = self.model.predict(next_state)[0]
        print(target, t, action)
        target[0][action] = reward + self.gamma * np.amax(t)
        self.model.fit(state, target, epochs=1, verbose=0)


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class Environment:
    def __init__(self):
        '''
        self.grid = np.array([
            [" ", " ", "X", " ", " ", " ", " ", " "],
            [" ", " ", "X", " ", " ", "X", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", "X", " ", " ", "X", " "],
            [" ", " ", " ", "X", " ", " ", "X", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", "END"]
        ])
        '''
        self.grid = sm
        self.agent_position = DEFAULT_AGNET_POS  # Starting position of the agent
        self.destination = (7, 3)  # Destination position

    def reset(self):
        self.agent_position = DEFAULT_AGNET_POS

    def step(self, action):
        if action == 0:  # Move up
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Move down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:  # Move left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:  # Move right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)

        # Check if the new position is valid
        if self._is_valid_move(new_position):
            self.agent_position = new_position

        # Determine reward and if the game is done
        reward = self._get_reward()
        done = self._is_done()

        return reward, done

    def _is_valid_move(self, position):
        row, col = position
        if 0 <= row < len(self.grid) and 0 <= col < len(self.grid[0]):
            return True
        return False

    def _get_reward(self):
        if self.grid[self.agent_position] == "END":
            return 100  # Reward for reaching the destination
        elif self.grid[self.agent_position] == "X":
            return -100  # Penalty for hitting an obstacle
        else:
            return -1  # Penalty for each move

    def _is_done(self):
        return self.grid[self.agent_position] == "END" or self.grid[self.agent_position] == "X"

    def render(self):
        rendered_grid = np.copy(self.grid)
        rendered_grid[self.agent_position] = "A"  # Render agent position
        print(rendered_grid)



def train():
    # Training parameters
    EPISODES = 1
    BATCH_SIZE = 32

    # Initialize environment and agent
    env = Environment()
    state_size = 2  # Assuming the agent's state representation is its (row, col) position
    action_size = 4  # Up, down, left, right
    agent = DQNAgent(state_size, action_size)


    try:
        agent.load("trained_model.h5")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")


    for episode in range(EPISODES):
        state = np.array(env.agent_position).reshape(1, -1)  # Initial state
        total_reward = 0
        done = False

        while not done:
            action = agent.pick_action(state)
            reward, done = env.step(action)
            total_reward += reward
            next_state = np.array(env.agent_position).reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)

        # Train the agent after each episode
        #if len(agent.memory) > BATCH_SIZE:
        #    agent.replay(BATCH_SIZE)
            agent.train(state, action, reward, next_state)
            state = next_state

        # Print episode results
        print(f"Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward}")
        env.reset()  # Reset the environment for the next episode

    # Save trained model
    agent.save("trained_model.h5")


def play():
    env = Environment()
    state_size = 2  # Assuming the agent's state representation is its (row, col) position
    action_size = 4  # Up, down, left, right
    agent = DQNAgent(state_size, action_size)


    try:
        agent.load("trained_model.h5")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")

    for episode in range(1):
        state = np.array(env.agent_position).reshape(1, -1)  # Initial state
        total_reward = 0
        done = False

        while not done:
            action = agent.pick_action(state)
            reward, done = env.step(action)
            total_reward += reward
            next_state = np.array(env.agent_position).reshape(1, -1)
            print("\n")
            print(action)
            env.render()
            state = next_state
        print("---------------------- END STATE ----------------------\n")
        env.render()
        print(f"Total Reward: {total_reward}")
        env.reset()  # Reset the environment for the next episode


usr = input("Train/Play (Enter: t or p): ")
if usr == "t":
    train()
elif usr == "p":
    play()