#!/usr/bin/env python3

import globals
import numpy as np
from tensorflow.keras.optimizers import Adam
from envs.one_wheelchair_env import OneWheelchairEnv
import sys

if __name__ == '__main__':
    env = OneWheelchairEnv()
    env.map_start_poimts = globals.map_start_poimts[globals.map]

    states = env.observation_space.shape
    actions = env.action_space.n
    
    model = globals.build_model(states, actions)
    #model.summary()
    dqn = globals.build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    if 'load' in sys.argv:
        dqn.load_weights(globals.weights_file_path)

    if 'train_nsteps' in sys.argv:
        dqn.fit(env, nb_steps=globals.train_steps, visualize=False, verbose=1)
        dqn.save_weights(globals.weights_file_path, overwrite=True)
    elif 'train_thresh' in sys.argv:
        env.reset_counters()
        while(globals.acc_thresh > ((env.success_episodes / env.total_episodes) if env.total_episodes else 0) or 
                globals.forward_movement_thresh > ((env.forward_steps / env.total_steps) if env.total_steps else 0)):
            dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)
            dqn.save_weights(globals.weights_file_path, overwrite=True)

            env.reset_counters()
            scores = dqn.test(env, nb_episodes=globals.test_episodes, visualize=False)
            print('Accurancy:', globals.acc_thresh, '/', env.success_episodes / env.total_episodes)
            print('Forward movements', globals.forward_movement_thresh, '/', env.forward_steps/env.total_steps)

    if 'test' in sys.argv:
        scores = dqn.test(env, nb_episodes=globals.test_episodes, visualize=False)
        print(np.mean(scores.history['episode_reward']))
