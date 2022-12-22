#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel, MoveModelRequest
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class Train1WheelchairEnv(Env):

    def __init__(self):
        #current data
        self.episode = 0
        self.action_n = 0
        self.max_episodes = 100
        self.split = 8
        self.lidar_sample = []
        self.collisions = False
        self.min_distance = 0.25
        self.finished = False
        
        #ros topics and services
        rospy.init_node('one_wheelchair_env', anonymous=True)

        self.scan_topic = "/static_laser1"   
        self.twist_topic = "/cmd_vel1"
        self.bumper_topic = "/collisions"
        self.prox_topic = "/prox_laser1"

        rospy.Subscriber(self.scan_topic, LaserScan, self.sample_lidar, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.bumper_topic, Collisions, self.check_collisions, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic, LaserScan, self.check_finished, buff_size=10000000, queue_size=1)

        rospy.wait_for_service("/move_model")
        self.move_model_service = rospy.ServiceProxy("/move_model", MoveModel)

        self.twist_pub = rospy.Publisher(self.twist_topic, Twist, queue_size=1)

        #learning env
        linear_speed = 0.5
        angular_speed = 1
        self.actions = [(0, 0), 
                        (linear_speed, 0), 
                        (-linear_speed, 0), 
                        (0, angular_speed), 
                        (0, -angular_speed), 
                        (linear_speed, angular_speed), 
                        (linear_speed, -angular_speed)]
        self.action_space = Discrete(len(self.actions))

        self.observation_space = Box(0, 5, shape=(1,self.split))
        while len(self.lidar_sample) == 0:pass
        self.state = np.array(self.lidar_sample)

    def step(self, action):
        self.move_robot(self.actions[action][0], self.actions[action][1])
        self.action_n += 1

        while self.episode == self.action_n: pass

        self.state = np.array(self.lidar_sample)
        
        done = True
        if self.collisions:
            reward = -200
        elif self.finished:
            reward = 100 + ((self.max_episodes - ((self.episode) * 100) / self.max_episodes))
        elif self.episode == self.max_episodes:
            reward = -200
        else:
            reward = 0
            done = False

        info = {}

        return self.state, reward, done, info
    
    def reset(self):
        self.lidar_sample = []
        pose = Pose2D()
        pose.x = 0.6
        pose.y = 0.5
        pose.theta = 1.57079632679
        self.move_model_service(name = "robot1", pose = pose)

        self.collisions = False
        self.finished = False
        self.episode = 0
        self.action_n = 0
        while len(self.lidar_sample) == 0:pass
        self.state = np.array(self.lidar_sample)

        return self.state


    def sample_lidar(self,data):
        self.move_robot(0,0)
        dist = [(len(data.ranges) // self.split) for _ in range(self.split)]
        for i in range(len(data.ranges) % self.split): dist[i] += 1  
        
        self.lidar_sample = []

        min_range =  len(data.ranges) - (dist[0] // 2)
        max_range = (dist[0] // 2) + (dist[0] % 2)
        self.lidar_sample.append(min(data.ranges[min_range:len(data.ranges)] + data.ranges[0:max_range]))
        
        for i in range(1, self.split):
            min_range = max_range
            max_range += dist[i]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))

        self.episode += 1

    def check_collisions(self, data):
        if len(data.collisions) > 0:
            self.move_robot(0,0)
            self.collisions = True

    def check_finished(self, data):
        if(min(data.ranges) < self.min_distance):
            self.move_robot(0,0)
            self.finished = True

    def move_robot(self, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        self.twist_pub.publish(twist_msg)



if __name__ == '__main__':
    env = Train1WheelchairEnv()
        
    #rospy.spin()        

    episodes = 1
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
            
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))

    states = env.observation_space.shape
    actions = env.action_space.n

    def build_model(states, actions):
        model = Sequential()    
        model.add(Dense(24, activation='relu', input_shape=states))
        model.add(Dense(24, activation='relu'))
        model.add(Flatten())
        model.add(Dense(actions, activation='linear'))
        return model
    
    model = build_model(states, actions)

    model.summary()

    def build_agent(model, actions):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=policy,  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
        return dqn

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    scores = dqn.test(env, nb_episodes=100, visualize=False)
    print(np.mean(scores.history['episode_reward']))

    _ = dqn.test(env, nb_episodes=15, visualize=True)
