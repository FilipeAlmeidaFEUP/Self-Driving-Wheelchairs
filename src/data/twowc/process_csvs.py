import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = ['straight']
no_test = ['straight']
maps = ['straight_hallway','left_door','center_door',
        'right_door','two_doors','small_obstacle',
        'big_obstacle','turn_left','turn_right',
        'curve_left','curve_right','full_turn_left',
        'full_turn_right','full_curve_left','full_curve_right']
offsets = [(0.725, 3.5),(2.225, 3.5),(3.725, 3.5),
            (5.225, 3.5),(6.725, 3.5),(8.225, 3.5),
            (9.725, 3.5),(2.5, 2.4),(0.6, 0.5),
            (5.5, 2.4),(3.6, 0.5),(8.07, 0.5),
            (6.93, 0.5),(11.4, 0.5),(9.6, 0.5)]

for file in files:
    data = pd.read_csv('train/' + file + '.csv')
    data = data.drop(data[data.task == 'train'].index)

    actions_per_test = [[],[],[],[],[]]
    n_steps = []
    reward_per_test = []
    end_conditions_per_test = [[], [], []]

    values = []

    i = 0
    
    for index, row in data.iterrows():
        if i%100 == 0:
            for action_type in actions_per_test: action_type.append(0)
            n_steps.append([])
            reward_per_test.append([])
            for end_condition_type in end_conditions_per_test: end_condition_type.append(0)

        actions = list(map(int, row['actions'].strip('][').split(', ')))
        for action in actions:
            if action == 2: actions_per_test[0][-1] += 1
            elif action == 0: actions_per_test[1][-1] += 1
            elif action == 3 or action == 4: actions_per_test[2][-1] += 1
            elif action == 1: actions_per_test[3][-1] += 1
            elif action == 5 or action == 6: actions_per_test[4][-1] += 1

        n_steps[-1].append(len(actions))

        rewards = list(map(float, row['rewards'].strip('][').split(', ')))
        reward_per_test[-1].append(sum(rewards))

        if row['end_condition'] == 'finished': end_conditions_per_test[0][-1] += 0.01
        elif row['end_condition'] == 'collision': end_conditions_per_test[1][-1] += 0.01
        elif row['end_condition'] == 'time out': end_conditions_per_test[2][-1] += 0.01

        i += 1
    
    #actions
    for index in range(len(actions_per_test)):actions_per_test[index] = tuple(actions_per_test[index])

    names = list(map(str, list(range(1, len(actions_per_test[0]) + 1))))

    fig = plt.figure(figsize=(9,5), dpi=200)
    left, bottom, width, height = 0.1, 0.3, 1.5, 0.6
    ax = fig.add_axes([left, bottom, width, height]) 
    
    width = 0.18   
    ticks = np.arange(len(names))    
    ax.bar(ticks - 0.36, actions_per_test[0], width, label='Back')
    ax.bar(ticks - 0.18, actions_per_test[1], width, align="center", label='Stop')
    ax.bar(ticks, actions_per_test[2], width, align="center", label='Rotate')
    ax.bar(ticks + 0.18, actions_per_test[3], width, align="center", label='Forward')
    ax.bar(ticks + 0.36, actions_per_test[4], width, align="center", label='Forward and Rotate')

    ax.set_ylabel('Action count')
    ax.set_xlabel('Test number')
    ax.set_xticks(ticks + width/5)
    ax.set_xticklabels(names)

    ax.legend(loc='best')
    plt.savefig('graphs/actions/' + file + '.png', bbox_inches = 'tight')

    plt.clf()

    #n_steps
    for i in range(len(n_steps)):
        n_steps[i] = sum(n_steps[i])/len(n_steps[i])

    fig = plt.figure(figsize=(6,5), dpi=200)
    left, bottom, width, height = 0.1, 0.3, 1.0, 0.6
    ax = fig.add_axes([left, bottom, width, height]) 

    ax.set_ylabel('Average number steps')
    ax.set_xlabel('Test number')

    plt.plot(list(range(1, len(n_steps) + 1)), n_steps)
    plt.savefig('graphs/n_steps/' + file + '.png', bbox_inches = 'tight')

    plt.clf()

    #rewards
    for i in range(len(reward_per_test)):
        reward_per_test[i] = sum(reward_per_test[i])/len(reward_per_test[i])

    fig = plt.figure(figsize=(6,5), dpi=200)
    left, bottom, width, height = 0.1, 0.3, 1.0, 0.6
    ax = fig.add_axes([left, bottom, width, height]) 

    ax.set_ylabel('Average reward')
    ax.set_xlabel('Test number')

    plt.plot(list(range(1, len(n_steps) + 1)), reward_per_test)
    plt.savefig('graphs/reward/' + file + '.png', bbox_inches = 'tight')

    plt.clf()

    #end conditions
    fig = plt.figure(figsize=(6,5), dpi=200)
    left, bottom, width, height = 0.1, 0.3, 1.0, 0.6
    ax = fig.add_axes([left, bottom, width, height]) 

    ax.set_ylabel('%')
    ax.set_xlabel('Test number')

    plt.plot(list(range(1, len(n_steps) + 1)), end_conditions_per_test[1], label = "Collision", alpha=0.2)
    plt.plot(list(range(1, len(n_steps) + 1)), end_conditions_per_test[2], label = "Time out", alpha=0.2)
    plt.plot(list(range(1, len(n_steps) + 1)), end_conditions_per_test[0], label = "Finished")
    plt.legend()
    plt.savefig('graphs/end_condition/' + file + '.png', bbox_inches = 'tight')

    plt.clf()

    #values
    values.append('Training:\n')
    values.append('Accuracy achieved: ' + str(end_conditions_per_test[0][-1]) + '\n')
    values.append('Trained for ' + str(len(n_steps) * 10000) + ' steps\n\n')

    with open('values/' + file + '.txt', 'w+') as f:
        f.writelines(values)
