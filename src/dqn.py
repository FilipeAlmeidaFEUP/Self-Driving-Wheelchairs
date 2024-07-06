#!/usr/bin/env python3

import globals
import numpy as np
from envs.one_wheelchair_env import OneWheelchairEnv
from envs.one_wheelchair_follow_wall_env import OneWheelchairFollowWallEnv
from envs.two_wheelchair_env_less_actions import TwoWheelChairEnvLessActions
from envs.one_wheelchair_follow_robot_env import OneWheelchairFollowRobotEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


'''

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
    dqn = DQNAgent(model=model, memory=memory, policy=policy,  nb_actions=actions, nb_steps_warmup=1000, target_model_update=1e-2)
    return dqn
'''

if __name__ == '__main__':
    if globals.use_two_wc:
        env = DummyVecEnv([lambda: TwoWheelChairEnvLessActions()])
    else:
        env = DummyVecEnv([lambda: OneWheelchairFollowRobotEnv()])
    env = env.envs[0]
    env.reset_robots()
    for map in globals.maps.items():
        if map[1]['usage']:
            env.map_names.append(map[0])
            env.start_points.append((map[0], map[1]['robot_space'], map[1]['end_space']))
            if ('overload_pos' in map[1].keys()):
                env.overload_pos[map[0]] = map[1]['overload_pos']

    '''
    states = env.observation_space.shape
    actions = env.action_space.n
    
    model = build_model(states, actions)
    print(model.summary())
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    '''

    if globals.load:
        model = DQN.load(globals.load_file, env=env)
    else:
        model = DQN('MlpPolicy', env, verbose=0, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, target_update_interval=100)

    if globals.train == 1:
        env.task = 'train'
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='dqn_model')
        model.learn(total_timesteps=globals.nsteps, callback=checkpoint_callback)
        if globals.save:
            model.save(globals.save_file)

    elif globals.train == 2:
        accuracies = []
        adjacencies = []
        nr_episodes = []

        eval_env = make_vec_env(lambda: env, n_envs=1)
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000, n_eval_episodes=globals.test_episodes, deterministic=True, render=False)

        model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, target_update_interval=50)#, exploration_fraction=0.2)
        env.reset_counters()
        max_acc = 0
        i=1
        while(globals.acc_thresh > ((env.success_episodes / env.total_episodes) if env.total_episodes else 0) or 
                globals.forward_movement_thresh > ((env.forward_steps / env.total_steps) if env.total_steps else 0) or
                0.8 > ((env.adjacent_episodes / env.total_episodes) if env.total_episodes else 0)):
            env.task = 'train'
            model.learn(total_timesteps=10000, callback=eval_callback)

            env.reset_counters()
            env.task = 'test'
            print('\n', 'Test number:', i)
            
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=globals.test_episodes)

            print("Mean reward:", mean_reward)
            print("Standard deviation of reward:", std_reward)

            acc = env.success_episodes / env.total_episodes
            print('Accurancy:', acc, '/', globals.acc_thresh)
            print('Forward movements', env.forward_steps / env.total_steps, '/', globals.forward_movement_thresh)
            print('Adjacency', env.adjacencies)
            if len(env.adjacencies) > 0:
                print('Adjacency average:', sum(env.adjacencies) / len(env.adjacencies))
            print('Adjacent episodes:', env.adjacent_episodes)
            if globals.save and acc > max_acc:
                max_acc = acc
                model.save(globals.save_file)
                if globals.save_data: env.dump_data(globals.data_file)
            i+=1
            
        accuracies.append(env.success_episodes / env.total_episodes)
        adjacencies.append(sum(env.adjacencies) / len(env.adjacencies))
        nr_episodes.append(i)
        print(f"Accuracies: {accuracies}")
        print(f"Adjacencies: {adjacencies}")
        print(f"Nr of train steps / 10000: {nr_episodes}")

    if globals.test:
        env.reset_counters()
        env.task = 'test'
        print("Evaluation:")
        print(evaluate_policy(model, env, n_eval_episodes=globals.test_episodes))
        env.success_episodes / env.total_episodes
        print('Accurancy:', env.success_episodes / env.total_episodes, '/', globals.acc_thresh)
        print('Forward movements', env.forward_steps / env.total_steps, '/', globals.forward_movement_thresh)
        print('Adjacency', env.adjacencies)
        if len(env.adjacencies) > 0:
            print('Adjacency average:', sum(env.adjacencies) / len(env.adjacencies))
        print('Adjacent episodes:', env.adjacent_episodes)

    if globals.save_data:
        env.dump_data(globals.data_file)

    env.reset_robots()
