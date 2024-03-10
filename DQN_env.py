import tensorflow as tf
from collections import deque
import numpy as np
import random
from obstacles import On_Off_Obstacle, Moving_Obstacle
from metric import Metrics
from grid import GridWorld
from util import encouragement
from util import env_to_vision


# TODO: Implement Double DQN (Have Model A for prediction, use Model B's value and Bellman algo to update Model A's values. Also implement batch logic where instead of only fitting 1 sample and target at a time, save the taget and samples in X, y pair and train it as a batch at the end)


  # Put the agent at this spot
DEFAULT_AGNET_POS = (3, 3)
DEFAULT_DES_POS = (11,11)
MAX_STEP = 80

gridMap = GridWorld()
gridMap.generate_grid_world()
gridMap.set_start_pos(DEFAULT_AGNET_POS[0], DEFAULT_AGNET_POS[1])
envMap = gridMap.grid



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
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.4  # exploration rate
        self.learning_rate = 0.001  # Lr
        self.model = self._build_model()    # Build the NN
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.swap_count = 0

    def _build_model(self):
        model = tf.keras.models.Sequential()    # A sequential NN
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu')) # Layer 1
        model.add(tf.keras.layers.Dense(64, activation='relu')) # Layer 2
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear')) # Layer 3
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    # Memory for experience replay
    def remember(self, state, action, reward, next_state, done):
        # Add the necessary information to memeory
        self.memory.append((state, action, reward, next_state, done))

    # Pick an action
    def pick_action(self, state):
        # If its smaller than epsilon, choose a random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Else, use the model to give you an action
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    # Train with experience replay
    def replay(self, batch_size):
        X = []
        y = []
        if self.swap_count == SWAP_COUNT:
            self.model.set_weights(self.target_model.get_weights())
            self.swap_count = 0
            #print('--------- MODEL SWAPPED ------------')
        else:
            self.swap_count += 1

    # Model A for prediction, use Model B's value and Bellman algo to update Model A's values
        minibatch = random.sample(self.memory, batch_size) # Sample a batch from memory
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)  # Get the model reading of the current state
            
            #print("target: ", target) 
            if done:    # If the state in the experience is done
                target[0][action] = reward      # Make the action that it took have the value of the reward
            else:
                t = self.model.predict(next_state, verbose=0)[0]   # If its not done, get the next state's Q values
                target[0][action] = reward + self.gamma * np.amax(t)    # Use bellman algo and put the result in the action it took
            
            X.append(state)
            y.append(target)
        
        cur_target = self.target_model.predict(state, verbose=0)
        cur_target[0][action] = reward + self.gamma * np.amax(cur_target)
        X.append(state)
        y.append(cur_target)
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[2])
        y = np.array(y)
        
        self.target_model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)  # Train

    # A function to just do normal training without expereicen replay
    def train(self, state, action, reward, next_state):
        target = self.model.predict(state, verbose=0)
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
        self.grid = envMap
        self.vision_grid = self.grid
        self.agent_position = DEFAULT_AGNET_POS  # Starting position of the agent
        self.destination = DEFAULT_DES_POS  # Destination position
        #self.state_size = np.prod(self.grid.shape)
        self.state_size = 9
        self.grid_mapping = {' ': 0, 'X': 1, 'END': 2, 'A': 3, '~': 0, 'E':2}  # Mapping for grid elements
        self.stepCount = 0
        self.temp_agent_pos = DEFAULT_AGNET_POS

    # Because the model can't take in strings, we have to convert the map to ints
    # This function converts the map back to strings for visual purpose
    def preprocess_state(self):
        self.vision_grid = self.state_to_vision()
        # processed_grid = np.vectorize(self.grid_mapping.get)(self.vision_grid)
        #return processed_grid.flatten().reshape(1, -1)
        return [self.vision_grid]

    # Reset Agent pos
    def reset(self):
        self.agent_position = DEFAULT_AGNET_POS
        self.temp_agent_pos = DEFAULT_AGNET_POS
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
            return encouragement(self.agent_position, self.temp_agent_pos, self.destination, False, self.stepCount, 1, -1) # Reward for each move

    def _is_done(self):
        return self.grid[self.agent_position] == "E" or self.grid[self.agent_position] == "X"

    def render(self):
        rendered_grid = np.copy(self.grid)
        rendered_grid[self.agent_position] = "A"  # Render agent position
        print(rendered_grid)

    def state_to_vision(self):
        return env_to_vision(self.grid, self.agent_position, False)


def train():
    EPISODES = 50
    BATCH_SIZE = 8

    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()


    try:
        agent.load("trained_model_env.h5")
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

            #r = random.randint(0,2)

            #print("\n--------------------------------")
            #print(state)
            #print(env.grid)
            #print("--------------------------------\n")

            reward, done = env.step(action)
            total_reward += reward
            next_state = env.preprocess_state()
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
            #agent.train(state, action, reward, next_state)
            state = next_state
            step += 1


        m.recordIteration(total_reward, True if total_reward > 0 else False, step)
        print(f"Episode: {episode + 1}/{EPISODES}, Total Reward: {total_reward}")
        env.reset()

    agent.save("trained_model_env.h5")
    m.printMetrics()


def play():
    env = Environment()
    state_size = env.state_size
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    m = Metrics()

    try:
        agent.load("trained_model_env.h5")
        print("Loaded model from disk")
    except:
        print("No pre-trained model found, starting training from scratch.")

    #print(env.grid)

    for episode in range(2):
        print("-------------- Start Iter ------------------")

        #o = On_Off_Obstacle(env.grid, 2, 3)
        #o = Moving_Obstacle(env.grid, 2, 3)
        #o.plot()

        #env.grid[env.agent_position] = 'A'
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
            print(env.grid)
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
