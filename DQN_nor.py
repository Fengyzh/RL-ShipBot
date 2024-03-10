import tensorflow as tf
from collections import deque
import numpy as np
import random
from obstacles import On_Off_Obstacle, Moving_Obstacle
from metric import Metrics
from grid import GridWorld
from util import encouragement


# TODO: Implement Double DQN (Have Model A for prediction, use Model B's value and Bellman algo to update Model A's values. Also implement batch logic where instead of only fitting 1 sample and target at a time, save the taget and samples in X, y pair and train it as a batch at the end)


  # Put the agent at this spot
DEFAULT_AGNET_POS = (7, 7)
DEFAULT_DES_POS = (11,11)
MAX_STEP = 100

gridMap = GridWorld()
gridMap.generate_grid_world()
gridMap.set_start_pos(DEFAULT_AGNET_POS[0], DEFAULT_AGNET_POS[1])



sm = np.array([        # The world
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

# Create Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size    # Size of the state (we taking the whole world)
        self.action_size = action_size  # Action size, we have up down left right so 4 actions
        self.memory = deque(maxlen=2000)    # Memory size for experience replay
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.3  # exploration rate
        self.learning_rate = 0.001  # Lr
        self.model = self._build_model()    # Build the NN
        self.swap_count = 0

    def _build_model(self):
        model = tf.keras.models.Sequential()    # A sequential NN
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu')) # Layer 1
        model.add(tf.keras.layers.Dense(64, activation='relu')) # Layer 2
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear')) # Layer 3
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    # Pick an action
    def pick_action(self, state):
        # If its smaller than epsilon, choose a random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Else, use the model to give you an action
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # A function to just do normal training without expereicen replay
    def train(self, state, action, reward, next_state):
        target = self.model.predict(state)
        t = self.model.predict(next_state)[0]
        target[0][action] = reward + self.gamma * np.amax(t)
        self.model.fit(state, target, epochs=1, verbose=0)

    # Load the model
    def load(self, name):
        self.model.load_weights(name)

    # Save the model
    def save(self, name):
        self.model.save_weights(name)

class Environment:
    def __init__(self):
        self.grid = gridMap.grid
        self.agent_position = DEFAULT_AGNET_POS  # Starting position of the agent
        self.destination = DEFAULT_DES_POS  # Destination position
        self.state_size = np.prod(self.grid.shape)
        self.grid_mapping = {' ': 0, 'X': 1, 'END': 2, 'A': 3, '~': 0, 'E':2}  # Mapping for grid elements
        self.stepCount = 0
        self.temp_agent_pos = DEFAULT_AGNET_POS

    # Because the model can't take in strings, we have to convert the map to ints
    # This function converts the map back to strings for visual purpose
    def preprocess_state(self):
        processed_grid = np.vectorize(self.grid_mapping.get)(self.grid)
        return processed_grid.flatten().reshape(1, -1)

    # Reset Agent pos
    def reset(self):
        self.agent_position = DEFAULT_AGNET_POS
        self.grid = self.grid.copy()

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
            return 100  # Reward for reaching the destination
        elif self.grid[self.agent_position] == "X":
            return -100  # Penalty for hitting an obstacle
        else:
            return encouragement(self.agent_position, self.temp_agent_pos, self.destination, False, self.stepCount) # Reward for each move

    def _is_done(self):
        return self.grid[self.agent_position] == "E" or self.grid[self.agent_position] == "X"

    def render(self):
        rendered_grid = np.copy(self.grid)
        rendered_grid[self.agent_position] = "A"  # Render agent position
        print(rendered_grid)



def train():
    EPISODES = 30
    BATCH_SIZE = 4

    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()


    try:
        agent.load("DQN_nor.h5")
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
            agent.train(state, action, reward, next_state)
            
            #agent.train(state, action, reward, next_state)
            state = next_state
            step += 1

        m.recordIteration(total_reward, True if total_reward > 0 else False, step)
        print(f"Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward}")
        env.reset()

    agent.save("DQN_nor.h5")
    m.printMetrics()


def play():
    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()

    try:
        agent.load("DQN_nor.h5")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")

    #print(env.grid)

    for episode in range(2):
        print("-------------- Start Iter ------------------")

        #o = On_Off_Obstacle(env.grid, 2, 3)
        #o = Moving_Obstacle(env.grid, 2, 3)
        #o.plot()

        env.grid[env.agent_position] = 'A'
        print(env.grid)

        state = env.preprocess_state()  # Initial state
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.pick_action(state)
            reward, done = env.step(action)
            total_reward += reward
            #o.tick()
            state = env.preprocess_state()
            #print(env.agent_position, o.get_obs_pos())
            #if env.agent_position == o.get_obs_pos():
                #done = True
                #total_reward -= 100

            print("\n")
            env.render()
            step += 1
        m.recordIteration(total_reward, True if env.agent_position == env.destination else False, step)
        print(f"Total Reward: {total_reward}")
        env.reset()
    m.printMetrics()


usr = input("Train/Play (Enter: t or p): ")
if usr == "t":
    train()
elif usr == "p":
    play()
