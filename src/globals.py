#!/usr/bin/env python3

#path to catkin workspace:
path_to_catkin = '/home/filipe/catkin_ws/'

#load wheights from file:
load = False
load_file = 'dqn_weights.h5f'
load_file = path_to_catkin + 'src/autowheelchairs_flatland/src/weights/' + load_file

#save wheights to file:
save = False
save_file = 'dqn_weights.h5f'
save_file = path_to_catkin + 'src/autowheelchairs_flatland/src/weights/' + save_file

#save data to file:
save_data = False
data_file = 'data.csv'
data_file = path_to_catkin + 'src/autowheelchairs_flatland/src/data/' + data_file

#dont train (0), train by number of steps (1), train by threshold (2)
train = 2
#if train by number of steps
nsteps = 50000
#if train by threshold
acc_thresh = 0.7
forward_movement_thresh = 0.8

#test the wheights
test = False
test_episodes = 100

#maps to use
maps = {
    'straight_hallway':{'usage':True, 'robot_space':(0.725, 3.5, 'h'), 'end_space':(0.725, 5.5, 'h')},
    'left_door':       {'usage':True, 'robot_space':(2.225, 3.5, 'h'), 'end_space':(2.225, 5.5, 'h')},
    'center_door':     {'usage':True, 'robot_space':(3.725, 3.5, 'h'), 'end_space':(3.725, 5.5, 'h')},
    'right_door':      {'usage':True, 'robot_space':(5.225, 3.5, 'h'), 'end_space':(5.225, 5.5, 'h')},
    'two_doors':       {'usage':True, 'robot_space':(6.725, 3.5, 'h'), 'end_space':(6.725, 5.5, 'h')},
    'small_obstacle':  {'usage':True, 'robot_space':(8.225, 3.5, 'h'), 'end_space':(8.225, 5.5, 'h')},
    'big_obstacle':    {'usage':True, 'robot_space':(9.725, 3.5, 'h'), 'end_space':(9.725, 5.5, 'h')},
    'turn_left':       {'usage':True, 'robot_space':(2.5, 2.4, 'v'), 'end_space':(0.6, 0.5, 'h')},
    'turn_right':      {'usage':True, 'robot_space':(0.6, 0.5, 'h'), 'end_space':(2.5, 2.4, 'v')},
    'curve_left':      {'usage':True, 'robot_space':(5.5, 2.4, 'v'), 'end_space':(3.6, 0.5, 'h')},
    'curve_right':     {'usage':True, 'robot_space':(3.6, 0.5, 'h'), 'end_space':(5.5, 2.4, 'v')},
    'full_turn_left':  {'usage':True, 'robot_space':(8.07, 0.5, 'h'), 'end_space':(6.93, 0.5, 'h')},
    'full_turn_right': {'usage':True, 'robot_space':(6.93, 0.5, 'h'), 'end_space':(8.07, 0.5, 'h')},
    'full_curve_left': {'usage':True, 'robot_space':(11.4, 0.5, 'h'), 'end_space':(9.6, 0.5, 'h')},
    'full_curve_right':{'usage':True, 'robot_space':(9.6, 0.5, 'h'), 'end_space':(11.4, 0.5, 'h')}
}
