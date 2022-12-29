#!/usr/bin/env python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

train_steps = 50000
test_episodes = 100
acc_thresh = 0.7
forward_movement_thresh = 0.8
path_to_catkin = ''
weights_file_name = 'dqn_weights.h5f'
weights_file_path = path_to_catkin + 'src/autowheelchairs_flatland/src/weights/' + weights_file_name

map = 5
map_start_poimts = [[(0.6, 0.5, 0.6, 2.5)], # 0 - straight hallway
                    [(0.6, 0.5, 0.6, 2.5),(1.85, 0.5, 1.85, 2.5),(3.1, 0.5, 3.1, 2.5)], # 1 - 3 straight hallways
                    [(0.6, 0.5, 0.6, 2.5),(1.85, 0.5, 1.85, 2.5),(3.1, 0.5, 3.1, 2.5),(4.35, 0.5, 4.35, 2.5),(5.6, 0.5, 5.6, 2.5),(6.85, 0.5, 6.85, 2.5)], # 2 - 6 straight hallways
                    [(0.6, 0.5, 2.5, 2.35)], # 3 - right turns
                    [(2.4, 0.5, 0.45, 2.35)], # 4 - left turns
                    [(0.6, 0.5, 2.5, 2.35),(5.4, 0.5, 3.5, 2.35)], # 5 - left and right turns
                    [(0.6, 0.5, 2.5, 2.35),(5.4, 0.5, 3.5, 2.35),(6.6, 0.5, 8.5, 2.35),(11.4, 0.5, 9.5, 2.35)]] # 6 - all turns

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,  nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn