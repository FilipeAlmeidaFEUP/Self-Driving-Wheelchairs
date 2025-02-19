import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel
from nav_msgs.msg import Odometry 
from gym import Env
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import numpy as np
import random
import pandas as pd 
from tf.transformations import euler_from_quaternion
import time

class TwoWheelChairEnvLessActions(Env):

    def __init__(self):
        #current data
        self.episode = 0
        self.steps = 0
        self.naction1 = 0
        self.naction2 = 0
        self.action_n = 0
        self.max_episodes = 300
        self.front_split = 9
        self.back_split = 3
        self.split = self.front_split + self.back_split
        
        self.lidar_sample = []
        self.lidar_sample2 = []
        
        self.collisions = False
        self.min_distance = 0.4
        
        self.finished = False
        self.end_reached = False
        self.finished2 = False
        self.end_reached2 = False
        
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0


        self.action_history2 = []
        self.rotation_counter2 = 0
        self.rotations2 = 0
        self.consecutive_rotations2 = 0

        self.forward_reward = 0

        
        
        self.map = 0
        self.start_points = []
        self.position = (0,0,0)
        self.position2 = (0,0,0)

        self.task = 'none'

        self.reset_counters()

        self.reset_data()
        
        #ros topics and services
        rospy.init_node('two_wheelchair_env', anonymous=True)

        self.scan_topic = "/static_laser1"   
        self.twist_topic = "/cmd_vel1"
        self.bumper_topic = "/collisions"
        self.odom_topic = "/odom1"
        self.prox_topic = "/prox_laser1"
        
        self.scan_topic2 = "/static_laser2"   
        self.twist_topic2 = "/cmd_vel2"
        self.odom_topic2 = "/odom2"
        self.prox_topic2 = "/prox_laser2"


        rospy.Subscriber(self.scan_topic, LaserScan, self.sample_lidar, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.bumper_topic, Collisions, self.check_collisions, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic, LaserScan, self.check_finished, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.update_position, buff_size=10000000, queue_size=1)
        
        rospy.Subscriber(self.scan_topic2, LaserScan, self.sample_lidar2, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic2, LaserScan, self.check_finished2, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.odom_topic2, Odometry, self.update_position2, buff_size=10000000, queue_size=1)

        rospy.wait_for_service("/move_model")
        self.move_model_service = rospy.ServiceProxy("/move_model", MoveModel)

        self.twist_pub = rospy.Publisher(self.twist_topic, Twist, queue_size=1)
        self.twist_pub2 = rospy.Publisher(self.twist_topic2, Twist, queue_size=1)


        #learning env
        linear_speed = 0.3
        angular_speed = 1.0471975512 
        self.actions = [(0, 0), 
                        (linear_speed, 0),  
                        (0, angular_speed)]
                        #(0, -angular_speed)]
                        
        n_actions = (len(self.actions) , len(self.actions))
                        
                        
        self.action_space = Discrete(np.prod(n_actions))


        self.observation_space = Box(0, 2, shape=(1, 2))
        
        
        while len(self.lidar_sample) != 1  or len(self.lidar_sample2) != 1 :pass
        self.state = np.array(self.lidar_sample + self.lidar_sample2)

    def step(self, action):
        while not (self.naction1 == self.steps + 1 and self.naction2 == self.steps + 1):
            time.sleep(0.00001)

        a1 = action // 3
        a2 = action % 3

        self.change_robot_speed(1, self.actions[a1][0], self.actions[a1][1])
        self.change_robot_speed(2, self.actions[a2][0], self.actions[a2][1])

        self.action_n += 1

        # OBS: It looks like this line tries to synchronize the current episode with the number of lidar readings
        #while self.episode == self.action_n: pass

        self.state = np.array(self.lidar_sample + self.lidar_sample2)
        
        done = False
        
        if self.steps < 4:
            self.collisions = False
            self.finished = False
            self.finished2 = False

        enter_end = False
        exit_end = False

        info = {}

        if self.end_reached != self.finished:
            if self.end_reached: exit_end = True
            else: enter_end = True
            self.end_reached = self.finished

        if self.end_reached2 != self.finished2:
            if self.end_reached2: exit_end = True
            else: enter_end = True
            self.end_reached2 = self.finished2

        if self.collisions:
            reward = -10 - self.forward_reward
            done = True
            self.data['end_condition'][-1] = 'collision'
        elif self.end_reached and self.end_reached2:
            reward = 10
            done = True
            self.data['end_condition'][-1] = 'finished'
        elif self.steps > self.max_episodes:
            reward = -10 - self.forward_reward
            done = True
            self.data['end_condition'][-1] = 'time out'
        else:  
            reward = 0
        
        if done:
            self.total_episodes += 1
            if self.end_reached and self.end_reached2: self.success_episodes += 1
            return self.state, reward, done, info

        if a1 == 1:
            reward += 1
            self.forward_reward += 1
        if a2 == 1:
            reward += 1
            self.forward_reward += 1
        
        if enter_end: reward += 10
        if exit_end: reward -= 10

        '''
        reward += self.action_reward(a1, 1)
        reward += self.action_reward(a2, 2)

        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            if (last_action == 2 and a1 == 3) or (last_action == 3 and a1 == 2):
                reward -= (300 / self.max_episodes) * 5

            last_action = self.action_history2[-1]
            if (last_action == 2 and a2 == 3) or (last_action == 3 and a2 == 2):
                reward -= (300 / self.max_episodes) * 5
            

        if a1 == 2: self.rotation_counter += 1 
        elif a1 == 3: self.rotation_counter -= 1
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

        if a2 == 2: self.rotation_counter2 += 1
        elif a2 == 3: self.rotation_counter2 -= 1
        if self.rotation_counter2 % 30 == 0:
            rot = self.rotation_counter2 / 30
            if self.rotations2 != rot:
                direction = rot - self.rotations2
                if self.consecutive_rotations2 != 0:
                    if direction != (self.consecutive_rotations2 / abs(self.consecutive_rotations2)):
                        self.consecutive_rotations2 = 0
                self.consecutive_rotations2 += direction
                self.rotations2 = rot
                reward -= 75 * abs(self.consecutive_rotations2)

        # Give reward if there is space in front of the robot
        reward += (self.lidar_sample[0] - 0.5) * (600 / self.max_episodes)
        # Give reward if there is space in front of the robot and to its front diagonals
        if self.lidar_sample[0] > 0.5: reward += ((self.lidar_sample[1] - 0.5) + (self.lidar_sample[2] - 0.5)) * (300 / self.max_episodes)

        reward += (self.lidar_sample2[0] - 0.5) * (600 / self.max_episodes)
        if self.lidar_sample2[0] > 0.5: reward += ((self.lidar_sample2[1] - 0.5) + (self.lidar_sample2[2] - 0.5)) * (300 / self.max_episodes)

        if max([self.lidar_sample[5], self.lidar_sample[6], self.lidar_sample2[5], self.lidar_sample2[6]]) <= 0.4 and abs(self.lidar_sample[5] - self.lidar_sample2[5]) <= 0.05 and abs(self.lidar_sample[6] - self.lidar_sample2[6]) <= 0.05 and a1 == 1 and a2 == 1:
            reward += 10
            self.data['adjacency'][-1].append(1)
            self.adj_steps += 1
        else: self.data['adjacency'][-1].append(0)

        '''

        self.total_steps += 1
        if a1 == 1:self.forward_steps += 0.5
        if a2 == 1:self.forward_steps += 0.5

        self.action_history.append(a1)
        self.action_history2.append(a2)

        self.data['actions'][-1].append(action)
        self.data['rewards'][-1].append(reward)
        self.data['positions'][-1].append((self.position, self.position2))

        assert self.naction1 == self.steps + 1
        assert self.naction2 == self.steps + 1

        # Updates the number of steps already taken
        self.steps += 1

        assert self.naction1 == self.steps
        assert self.naction2 == self.steps

        return self.state, reward, done, info
    
    def action_reward(self, action, chair):
        if (chair == 1 and self.end_reached) or (chair == 2 and self.end_reached2):
            if action == 0: return 300 / self.max_episodes
            else: return -300 / self.max_episodes

        if action == 1:
            reward = 300 / self.max_episodes
            self.forward_reward += reward
        elif action == 0:
            reward = -(300 / self.max_episodes) * 8
        elif action == 2 or action == 3:
            if chair == 1 and self.lidar_sample[0] > 0.1: reward = -(300 / self.max_episodes) * 2
            elif chair == 2 and self.lidar_sample2[0] > 0.1: reward = -(300 / self.max_episodes) * 2
            else:reward = 300 / (self.max_episodes * 2)
        else:
            reward = 0
        return reward
    
    def render(self): pass
    
    def reset(self):
        self.lidar_sample = []
        self.lidar_sample2 = []

        map = self.start_points[self.map]

        x = map[1][0]
        y = map[1][1]

        x2 = map[1][0]
        y2 = map[1][1]
        theta = random.random() * (math.pi * 2)
       
        if map[1][2] == 'h': 
            x -= 0.2
            x2 += 0.2
            theta = math.pi / 2
        else: 
            y -= 0.2
            y2 += 0.2
            theta = math.pi
        self.change_robot_position("robot1", x, y, theta)
        self.change_robot_position("robot2", x2, y2, theta)


        target_x = map[2][0]
        target_y = map[2][1]
        self.change_robot_position("prox", target_x, target_y, 0)

        self.collisions = False
        self.finished = False
        self.finished2 = False
        self.end_reached = False
        self.end_reached2 = False
        self.episode = 0
        self.action_n = 0
        self.steps = 0
        self.naction1 = 0
        self.naction2 = 0
        if hasattr(self, "last_execution_time"):
            delattr(self, "last_execution_time")
        if hasattr(self, "last_execution_time2"):
            delattr(self, "last_execution_time2")

        while len(self.lidar_sample) != 1 or len(self.lidar_sample2) != 1:pass

        self.state = np.array(self.lidar_sample + self.lidar_sample2)

        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0

        self.action_history2 = []
        self.rotation_counter2 = 0
        self.rotations2 = 0
        self.consecutive_rotations2 = 0

        self.forward_reward = 0

        self.map += 1
        if self.map == len(self.start_points): self.map = 0

        self.data['map'].append(map[0])
        self.data['task'].append(self.task)
        self.data['target'].append((target_x,target_y))
        self.data['actions'].append([])
        self.data['rewards'].append([])
        self.data['positions'].append([(x,y,theta)])
        self.data['adjacency'].append([])
        self.data['end_condition'].append('time out')

        return self.state

    def sample_lidar(self,data):
        if self.naction1 != self.steps: return

        current_time = time.time()
        if hasattr(self, 'last_execution_time'):
            elapsed_time = current_time - self.last_execution_time
            if elapsed_time < 0.5:
                return
        else:
            self.last_execution_time = current_time

        self.change_robot_speed(1,0,0)

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
        '''
        
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) + (each // 2)]))
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 4) - 1]))
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) * (3/4)) - 1]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 4) - 1) - (each // 2)]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 4) - 1) + (each // 2)]))


        back_lasers = len(data.ranges) - front_lasers
        back_dist = [(back_lasers // self.back_split) for _ in range(self.back_split)]
        for i in range(back_lasers % self.back_split): back_dist[i] += 1

        dist = back_dist[:math.ceil(self.back_split/2)] + front_dist + back_dist[math.ceil(self.back_split/2):]

        if self.back_split % 2 == 0:
            min_range = 0
            max_range = dist[0]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))
        else:
            min_range = len(data.ranges) - (dist[0] // 2)
            max_range = (dist[0] // 2) + (dist[0] % 2)
            self.lidar_sample.append(min(data.ranges[min_range:len(data.ranges)] + data.ranges[0:max_range]))
        
        for i in range(1, self.split):
            min_range = max_range
            max_range += dist[i]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))

        for i in range(len(self.lidar_sample)): self.lidar_sample[i] = min(self.lidar_sample[i], 2)
        '''

        # TODO: check this line again
        self.episode += 0.5

        assert self.naction1 == self.steps

        # Updates number of lidar readings from robot 1
        self.naction1 += 1

        assert self.naction1 == self.steps + 1

    def sample_lidar2(self,data):
        if self.naction2 != self.steps: return

        current_time2 = time.time()
        if hasattr(self, 'last_execution_time2'):
            elapsed_time2 = current_time2 - self.last_execution_time2
            if elapsed_time2 < 0.5:
                return
        else:
            self.last_execution_time2 = current_time2

        self.change_robot_speed(2,0,0)

        self.lidar_sample2 = []

        front_lasers = math.ceil(len(data.ranges) / 2)
        each = front_lasers // (self.front_split - 1)
        front_lasers += each
        front_dist = [each for _ in range(self.front_split)]
        back = False
        for i in range(len(data.ranges) % self.split): 
            if back: front_dist[self.front_split - ((i//2)+1)] += 1
            else: front_dist[i//2] += 1
            back = not back
        
        self.lidar_sample2.append((data.ranges[math.ceil(len(data.ranges) / 2) - 1]))
        '''
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample2.append((data.ranges[math.ceil(len(data.ranges) / 4) - 1]))
        self.lidar_sample2.append((data.ranges[math.ceil(len(data.ranges) * (3/4)) - 1]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) * (3/4)) - 1) + (each // 2)]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) * (3/4)) - 1) - (each // 2)]))


        back_lasers = len(data.ranges) - front_lasers
        back_dist = [(back_lasers // self.back_split) for _ in range(self.back_split)]
        for i in range(back_lasers % self.back_split): back_dist[i] += 1

        dist = back_dist[:math.ceil(self.back_split/2)] + front_dist + back_dist[math.ceil(self.back_split/2):]

        if self.back_split % 2 == 0:
            min_range =  0
            max_range = dist[0]
            self.lidar_sample2.append(min(data.ranges[min_range:max_range]))
        else:
            min_range =  len(data.ranges) - (dist[0] // 2)
            max_range = (dist[0] // 2) + (dist[0] % 2)
            self.lidar_sample2.append(min(data.ranges[min_range:len(data.ranges)] + data.ranges[0:max_range]))
        
        for i in range(1, self.split):
            min_range = max_range
            max_range += dist[i]
            self.lidar_sample2.append(min(data.ranges[min_range:max_range]))

        for i in range (len(self.lidar_sample2)): self.lidar_sample2[i] = min(self.lidar_sample2[i], 2)
        '''

        # TODO: Check this line again
        self.episode += 0.5

        assert self.naction2 == self.steps
        # Updates number of lidar readings from robot 2
        self.naction2 += 1
        assert self.naction2 == self.steps + 1

    def check_collisions(self, data):
        if self.collisions: return
        if len(data.collisions) > 0:
            self.change_robot_speed(1,0,0)
            self.change_robot_speed(2,0,0)

            self.collisions = True

    def check_finished(self, data):
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            self.finished = min(values) < self.min_distance


    def check_finished2(self, data):
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            self.finished2 = min(values) < self.min_distance
                
    def update_position(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

    def update_position2(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position2 = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

    def change_robot_speed(self, robot, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular

        if(robot == 1):
            self.twist_pub.publish(twist_msg)
        if(robot == 2):
            self.twist_pub2.publish(twist_msg)

    def reset_robots(self):
        self.change_robot_position('robot1', 11.25, 4.5, 1.57079632679)
        self.change_robot_position('robot2', 11.25, 4.2, 1.57079632679)
        self.change_robot_position('prox', 11.25, 4.8, 0)

    def change_robot_position(self, name, x, y, theta):
        pose = Pose2D()
        pose.x = x
        pose.y = y
        pose.theta = theta
        self.move_model_service(name = name, pose = pose)

    def reset_counters(self):
        self.total_episodes = 0
        self.success_episodes = 0
        self.total_steps = 0
        self.forward_steps = 0
        self.adj_steps = 0

    def reset_data(self):
        self.data = {
            'map':[],
            'task':[],
            'target':[],
            'actions':[],
            'rewards':[],
            'positions':[],
            'adjacency':[],
            'end_condition':[]
        }
    
    def dump_data(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename)


