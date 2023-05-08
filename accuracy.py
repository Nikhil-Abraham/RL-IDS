import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import random
import gym
from gym import spaces
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Load the dataset
df = pd.read_csv("./Data/new2.csv")

# Drop irrelevant columns (such as timestamp)
df = df.drop(columns=["Timestamp"])

# Drop columns with same values throughout
dropped_cols = list(df.columns[df.nunique() == 1])
df = df.drop(columns=dropped_cols)

# Replace infinite or very large values with the maximum of the corresponding feature
df = df.replace([np.inf, -np.inf], np.nan)
max_values = df.max()
df = df.fillna(max_values)

# Select only the relevant features
selected_features = ['Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
X = df[selected_features].values

# Define the environment
class IDS(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data):
        super(IDS, self).__init__()
        self.data = data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(selected_features),))
        self.action_space = spaces.Discrete(2)
        self.current_step = 0
        self.max_steps = len(self.data) - 1
    
    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step == self.max_steps
        reward = 1 if action == 1 and self.data[self.current_step][0] > 0 else 0
        return self.data[self.current_step], reward, done, {}
    
    def render(self, mode='human', close=False):
        pass

# Create the environment
env = IDS(X)

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

# Define the memory
memory = SequentialMemory(limit=10000, window_length=1)

# Define the policy
policy = BoltzmannQPolicy()

# Define the agent
dqn = DQNAgent(model=model, 
               nb_actions=env.action_space.n, 
               memory=memory, 
               nb_steps_warmup=10, 
               target_model_update=1e-2, 
               policy=policy)

# Compile the agent
dqn.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae', 'accuracy'])


# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Evaluate the agent
scores = dqn.test(env, nb_episodes=2, visualize=False)


# Print the results
print('Scale of rewards:', '[0, 1]')
print('Mean test reward:', np.mean(scores.history['episode_reward']))



