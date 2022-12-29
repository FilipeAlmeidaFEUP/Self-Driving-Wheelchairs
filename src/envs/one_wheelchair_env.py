import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np


class OneWheelchairEnv(Env):

    def __init__(self):
        #current data
        self.episode = 0
        self.action_n = 0
        self.max_episodes = 150
        self.front_split = 9
        self.back_split = 3
        self.split = self.front_split + self.back_split
        self.lidar_sample = []
        self.collisions = False
        self.min_distance = 0.25
        self.finished = False
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0
        self.forward_reward = 0
        self.map = 0
        self.map_start_poimts = []

        self.reset_counters()
        
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
        linear_speed = 0.3
        angular_speed = 1.0471975512 
        self.actions = [(0, 0), 
                        (linear_speed, 0), 
                        (-linear_speed, 0), 
                        (0, angular_speed), 
                        (0, -angular_speed), 
                        (linear_speed, angular_speed), 
                        (linear_speed, -angular_speed)]
        self.action_space = Discrete(len(self.actions))

        self.observation_space = Box(0, 2, shape=(1,self.split + 5))
        while len(self.lidar_sample) == 0:pass
        self.state = np.array(self.lidar_sample)

    def step(self, action):
        
        self.move_robot(self.actions[action][0], self.actions[action][1])
        self.action_n += 1

        while self.episode == self.action_n: pass

        self.state = np.array(self.lidar_sample)
        
        done = False
        
        if self.action_n < 4:
            self.collisions = False
            self.finished = False

        if self.collisions:
            reward = -200
            done = True
        elif self.finished:
            reward = 400 + ((self.max_episodes - ((self.episode) * 200) / self.max_episodes))
            done = True
        elif self.episode == self.max_episodes:
            reward = -(300 + self.forward_reward) 
            done = True
        elif action == 1:
            reward = 300 / self.max_episodes
            self.forward_reward += reward
        elif action == 5 or action == 6:
            reward = 300 / (self.max_episodes * 2)
            self.forward_reward += reward
        elif action == 0 or action == 2:
            reward = -(300 / self.max_episodes) * 5
        elif action == 3 or action == 4:
            reward = -(300 / self.max_episodes) * 2
        else:
            reward = 0

        if len(self.action_history) > 0:
            last_action = self.action_history[len(self.action_history)-1]
            if (last_action == 1 and action == 2) or (last_action == 2 and action == 1):
                reward -= (300 / self.max_episodes) * 5
            elif (last_action == 3 and action == 4) or (last_action == 4 and action == 3):
                reward -= (300 / self.max_episodes) * 5

        if action == 3 or action == 5: self.rotation_counter += 1 
        elif action == 4 or action == 6: self.rotation_counter -= 1
        if self.rotation_counter % 30 == 0:
            rot = self.rotation_counter / 30
            if self.rotations != rot:
                direction = rot - self.rotations
                if self.consecutive_rotations != 0:
                    if direction != (self.consecutive_rotations / abs(self.consecutive_rotations)):
                        self.consecutive_rotations = 0
                self.consecutive_rotations += direction
                self.rotations = rot
                reward -= 75 * abs(self.consecutive_rotations)
        
        reward += (self.lidar_sample[0] - 0.5) * 2
        if self.lidar_sample[0] > 0.5: reward += (self.lidar_sample[1] - 0.5) + (self.lidar_sample[2] - 0.5)

        #if done: print("  -->", reward, self.episode)

        self.total_steps += 1
        if action in [1,5,6]:self.forward_steps += 1
        if done:
            self.total_episodes += 1
            if self.finished: self.success_episodes += 1

        info = {}

        self.action_history.append(action)

        return self.state, reward, done, info
    
    def render(self): pass
    
    def reset(self):
        self.lidar_sample = []
        pose = Pose2D()
        pose.x = self.map_start_poimts[self.map][0]
        pose.y = self.map_start_poimts[self.map][1]
        pose.theta = 1.57079632679
        self.move_model_service(name = "robot1", pose = pose)
        pose.x = self.map_start_poimts[self.map][2]
        pose.y = self.map_start_poimts[self.map][3]
        self.move_model_service(name = "prox", pose = pose)

        self.collisions = False
        self.finished = False
        self.episode = 0
        self.action_n = 0
        while len(self.lidar_sample) == 0:pass
        self.state = np.array(self.lidar_sample)
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0
        self.forward_reward = 0

        self.map += 1
        if self.map == len(self.map_start_poimts): self.map = 0

        return self.state


    def sample_lidar(self,data):
        self.move_robot(0,0)

        self.lidar_sample = []

        front_lasers = math.ceil(len(data.ranges) / 2)
        each = front_lasers // (self.front_split - 1)
        front_lasers += each
        front_dist = [each for _ in range(self.front_split)]
        back = False
        for i in range(len(data.ranges) % self.split): 
            if back: front_dist[self.front_split - ((i//2)+1)] += 1
            else: front_dist[i//2] += 1
            back = not back
        
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 2) - 1]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 4) - 1]))
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) * (3/4)) - 1]))

        back_lasers = len(data.ranges) - front_lasers
        back_dist = [(back_lasers // self.back_split) for _ in range(self.back_split)]
        for i in range(back_lasers % self.back_split): back_dist[i] += 1

        dist = back_dist[:math.ceil(self.back_split/2)] + front_dist + back_dist[math.ceil(self.back_split/2):]

        if self.back_split % 2 == 0:
            min_range =  0
            max_range = dist[0]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))
        else:
            min_range =  len(data.ranges) - (dist[0] // 2)
            max_range = (dist[0] // 2) + (dist[0] % 2)
            self.lidar_sample.append(min(data.ranges[min_range:len(data.ranges)] + data.ranges[0:max_range]))
        
        for i in range(1, self.split):
            min_range = max_range
            max_range += dist[i]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))

        for i in range (len(self.lidar_sample)): self.lidar_sample[i] = min(self.lidar_sample[i], 2)

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

    def reset_counters(self):
        self.total_episodes = 0
        self.success_episodes = 0
        self.total_steps = 0
        self.forward_steps = 0