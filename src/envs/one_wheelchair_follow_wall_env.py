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

class OneWheelchairFollowWallEnv(Env):

    def __init__(self):
        #current data
        self.episode = 0
        self.steps = 0
        self.naction1 = 0
        self.action_n = 0
        self.max_episodes = 300
        self.front_split = 9
        self.back_split = 3
        self.split = self.front_split + self.back_split
        self.ideal_dist = 0.4
        self.ideal_angle = math.pi/2
        self.total_episodes = 0
        self.total_reward = 0
        
        self.lidar_sample = []
        
        self.collisions = False
        self.min_distance = 0.4
        
        self.finished = False
        self.end_reached = False
        
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0
        self.forward_reward = 0

        
        
        self.map = 0
        self.start_points = []
        self.position = (0,0,0)

        self.task = 'none'

        self.reset_counters()

        self.reset_data()
        
        #ros topics and services
        rospy.init_node('one_wheelchair_env_wall', anonymous=True)

        self.scan_topic = "/static_laser1"   
        self.twist_topic = "/cmd_vel1"
        self.bumper_topic = "/collisions"
        self.odom_topic = "/odom1"
        self.prox_topic = "/prox_laser1"
        
        rospy.Subscriber(self.scan_topic, LaserScan, self.sample_lidar, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.bumper_topic, Collisions, self.check_collisions, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic, LaserScan, self.check_finished, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.update_position, buff_size=10000000, queue_size=1)
        
        rospy.wait_for_service("/move_model")
        self.move_model_service = rospy.ServiceProxy("/move_model", MoveModel)

        self.twist_pub = rospy.Publisher(self.twist_topic, Twist, queue_size=1)

        #learning env
        linear_speed = 0.3
        angular_speed = 0.5
        # TODO: divide case with 0 angular speed to 2 with low angular speed
        self.actions = [(linear_speed, -angular_speed),
                        (linear_speed, 0),
                        (linear_speed, angular_speed)]
                        
        n_actions = (len(self.actions))
                        
        self.action_space = Discrete(n_actions)

        self.observation_space = Box(low=np.array([[0.0, 0.0, 0.0]]), high=np.array([[10.0, math.pi, 0.0]]), shape=(1, 3))
        
        while len(self.lidar_sample) != 46: pass

        # TODO: add front, side and back rays. Maybe a sample of the rays
        self.state = np.array(self.get_min_dist_ray(self.lidar_sample))


    def step(self, action):
        while not (self.naction1 == self.steps + 1):
            time.sleep(0.00001)

        reward = 0

        self.change_robot_speed(1, self.actions[action][0], self.actions[action][1])

        self.action_n += 1

        # OBS: It looks like this line tries to synchronize the current episode with the number of lidar readings
        #while self.episode == self.action_n: pass

        self.state = np.array(self.get_min_dist_ray(self.lidar_sample))
        
        done = False
        
        if self.steps < 4:
            self.collisions = False
            self.finished = False

        info = {}

        if self.collisions:
            reward = -1
            done = True
            self.total_episodes += 1
            self.data['end_condition'][-1] = 'collision'
            self.total_reward = 0
        elif self.finished:
            # TODO: Maybe lower this reward
            done = True
            self.total_episodes += 1
            self.success_episodes += 1
            self.data['end_condition'][-1] = 'finished'
            self.total_reward = 0
        elif self.episode > self.max_episodes:
            done = True
            self.total_episodes += 1
            self.data['end_condition'][-1] = 'time out'
            self.total_reward = 0
        else:
            min_dist, min_angle, _ = self.state
            # TODO: se ambas as condições estiverem no intervalo definido reward += 1
            safe_angle = 20*math.pi/180
            # dist_reward = max(0, 0.2 - (abs(min_dist - self.ideal_dist))) / 0.4
            dist_reward = 0
            angle_reward = 0
            if 0.2 - (abs(min_dist - self.ideal_dist)) < 0:
                dist_reward = -0.1
            # angle_reward = max(0, (safe_angle) - (abs(min_angle - self.ideal_angle))) / (safe_angle*2)
            if (safe_angle) - (abs(min_angle - self.ideal_angle)) < 0:
                angle_reward = -0.1
            reward = dist_reward + angle_reward
            self.total_reward += reward


        self.total_steps += 1

        self.data['actions'][-1].append(action)
        self.data['rewards'][-1].append(reward)
        self.data['positions'][-1].append(self.position)

        assert self.naction1 == self.steps + 1

        # Updates the number of steps already taken
        self.steps += 1

        assert self.naction1 == self.steps

        return self.state, reward, done, info
    
    def get_min_dist_ray(self, lidar_data):
        n_rays = len(lidar_data)
        delta_ang = math.pi / n_rays
        min_dist = float('inf')
        min_angle = float('inf')

        for i in range(len(lidar_data)):
            if lidar_data[i] < min_dist:
                min_dist = lidar_data[i]
                min_angle = i*delta_ang
        
        return min_dist, min_angle, lidar_data[0]
    
    def render(self): pass
    
    def reset(self):
        self.lidar_sample = []

        map = self.start_points[self.map]

        x = map[1][0]
        y = map[1][1]

        theta = math.pi / 2 - 0.4
       
        self.change_robot_position("robot1", x, y, theta)

        target_x = map[2][0]
        target_y = map[2][1]
        self.change_robot_position("prox", target_x, target_y, 0)

        self.collisions = False
        self.finished = False
        self.end_reached = False
        self.episode = 0
        self.action_n = 0
        self.steps = 0
        self.naction1 = 0
        self.total_reward = 0
        if hasattr(self, "last_execution_time"):
            delattr(self, "last_execution_time")

        while len(self.lidar_sample) != 46: pass

        self.state = np.array(self.get_min_dist_ray(self.lidar_sample))

        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0

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

        self.lidar_sample = data.ranges

        self.episode += 0.5 

        assert self.naction1 == self.steps

        # Updates number of lidar readings from robot 1
        self.naction1 += 1

        assert self.naction1 == self.steps + 1

    def check_collisions(self, data):
        if self.collisions: return
        if len(data.collisions) > 0:
            self.change_robot_speed(1,0,0)

            self.collisions = True

    def check_finished(self, data):
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            self.finished = min(values) < self.min_distance


    def update_position(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

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
        self.change_robot_position('robot2', 22.25, 4.2, 1.57079632679)
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


