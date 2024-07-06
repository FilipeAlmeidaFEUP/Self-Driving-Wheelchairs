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
from stable_baselines3 import DQN

class OneWheelchairFollowRobotEnv(Env):

    def __init__(self):
        self.overload_pos = {}
        self.map_names = []

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
        self.ideal_dist_wall = 0.3
        self.ideal_dist = 0.4
        self.ideal_angle = math.pi/2
        self.total_episodes = 0
        self.total_reward = 0
        self.last_dist_diff = 0
        
        self.lidar_sample = []
        self.lidar_sample2 = []

        self.wallFollowState = None
        
        self.collisions = False
        self.min_distance = 0.4
        self.finished = False
        self.end_reached = False
        
        self.finished2 = False
        
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0
        self.forward_reward = 0

        self.point_cloud = []
        self.clusters = []     
        
        self.map = 0
        self.start_points = []
        self.position = (0,0,0)

        self.position2 = (0,0,0)

        self.task = 'none'

        self.wallFollowModel = DQN.load("/home/teste/work/catkin_ws/src/autowheelchairs_flatland/src/weights/wall_follower_90.h5f")

        self.reset_counters()

        self.reset_data()
        
        #ros topics and services
        rospy.init_node('one_wheelchair_env_robot_follow', anonymous=True)

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
        angular_speed = 0.5
        # TODO: divide case with 0 angular speed to 2 with low angular speed
        self.actions = [
            (linear_speed, angular_speed),
            (linear_speed, 0),
            (0, 0),
            (linear_speed, -angular_speed)
        ]
        
        self.wallFollowActions = [
            (0.2, -angular_speed),
            (0.2, 0),
            (0.2, angular_speed)
        ]

        n_actions = (len(self.actions))
                        
        self.action_space = Discrete(n_actions)

        self.observation_space = Box(low=np.array([0.0, 0.0]), high=np.array([10.0, math.pi]), shape=(2,), dtype=np.float64)

        while len(self.lidar_sample) != 46 or len(self.lidar_sample2) != 46: pass

        # TODO: add front, side and back rays. Maybe a sample of the rays
        min_dist, angle = self.min_dist_robot(0.1)
        self.state = np.array((min_dist - self.ideal_dist, angle))
        self.last_dist_diff = min_dist - self.ideal_dist

        min_dist2, angle2 = self.get_min_dist_ray(self.lidar_sample2)
        self.wallFollowState = np.array((min_dist2 - self.ideal_dist_wall, angle2))


    def step(self, action):
        while not (self.naction1 == self.steps + 1 and self.naction2 == self.steps + 1):
            time.sleep(0.00001)

        reward = 0

        self.change_robot_speed(1, self.actions[action][0], self.actions[action][1])

        min_dist2, angle2 = self.get_min_dist_ray(self.lidar_sample2)
        self.wallFollowState = np.array((min_dist2 - self.ideal_dist_wall, angle2))

        wallFollowAction, _ = self.wallFollowModel.predict(self.wallFollowState)

        self.change_robot_speed(2, self.wallFollowActions[wallFollowAction][0], self.wallFollowActions[wallFollowAction][1])

        self.action_n += 1

        # OBS: It looks like this line tries to synchronize the current episode with the number of lidar readings
        #while self.episode == self.action_n: pass

        min_dist, angle = self.min_dist_robot(0.1)
        self.state = np.array((min_dist - self.ideal_dist, angle))

        self.last_dist_diff = min_dist - self.ideal_dist

        done = False
        
        if self.steps < 4:
            self.collisions = False
            self.finished = False

        info = {}

        if self.collisions:
            reward = -3
            done = True
            self.total_episodes += 1
            self.data['end_condition'][-1] = 'collision'
            self.total_reward = 0
            self.adjacencies.append(self.adj_steps / self.total_steps)

        elif self.finished and self.finished2:
            done = True
            self.total_episodes += 1
            self.success_episodes += 1

            if self.adj_steps / self.total_steps > 0.80:
                self.adjacent_episodes += 1

            self.data['end_condition'][-1] = 'finished'
            self.total_reward = 3
            self.adjacencies.append(self.adj_steps / self.total_steps)
        elif self.episode > self.max_episodes:
            reward = -1
            done = True
            self.total_episodes += 1
            self.data['end_condition'][-1] = 'time out'
            self.total_reward = 0
            self.adjacencies.append(self.adj_steps / self.total_steps)

        else:
            # TODO: se ambas as condições estiverem no intervalo definido reward += 1
            safe_angle = 50*math.pi/180
            # dist_reward = max(0, 0.2 - (abs(min_dist - self.ideal_dist))) / 0.4
            dist_reward = 0
            angle_reward = 0
            if 0.1 - (abs(min_dist - self.ideal_dist)) < 0:
                dist_reward = -0.1
            # angle_reward = max(0, (safe_angle) - (abs(min_angle - self.ideal_angle))) / (safe_angle*2)
            if (safe_angle) - (abs(angle - self.ideal_angle)) < 0:
                angle_reward = -0.1
            if dist_reward == 0 and angle_reward == 0:
                reward = 0.1
            else:
                reward = dist_reward + angle_reward
            
            if 0.15 - (abs(min_dist - self.ideal_dist)) >= 0 and (safe_angle) - (abs(angle - self.ideal_angle)) >= 0:
                self.adj_steps += 1
            
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
        
        return min_dist, min_angle
    
    def render(self): pass
    
    def reset(self):
        self.lidar_sample = []
        self.lidar_sample2 = []

        map = self.start_points[self.map]

        random_offset = random.uniform(-0.05, 0.05)

        if map[1][2] == 'v':
            x2 = map[1][0] + 0.1
            y2 = map[1][1] - 0.2 + random_offset

            x = map[1][0] 
            y = map[1][1] + 0.2 + random_offset
            theta = math.pi
        else:
            x2 = map[1][0] - 0.2 + random_offset
            y2 = map[1][1]

            x = map[1][0] + 0.2 + random_offset
            y = map[1][1]
            theta = math.pi / 2

        if self.map_names[self.map] in self.overload_pos.keys():
            x, y, x2, y2, theta = self.overload_pos[self.map_names[self.map]]
       
        self.change_robot_position("robot2", x2, y2, theta)
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
        self.naction2 = 0
        self.total_reward = 0

        while len(self.lidar_sample) != 46: pass
        while len(self.lidar_sample2) != 46: pass

        min_dist, angle = self.min_dist_robot(0.1)
        self.state = np.array((min_dist - self.ideal_dist, angle))

        min_dist2, angle2 = self.get_min_dist_ray(self.lidar_sample2)
        self.wallFollowState = np.array((min_dist2 - self.ideal_dist_wall, angle2))

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

    def euclid_dist(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

    def sample_lidar(self, data):
        if self.naction1 != self.steps: return

        self.change_robot_speed(1,0,0)

        self.lidar_sample = data.ranges

        self.point_cloud = self.get_point_cloud()

        self.clusters = self.get_point_cloud_clusters(0.2)

        self.episode += 0.5 

        assert self.naction1 == self.steps

        # Updates number of lidar readings from robot 1
        self.naction1 += 1

        assert self.naction1 == self.steps + 1

    def sample_lidar2(self,data):
        if self.naction2 != self.steps: return
        self.change_robot_speed(2,0,0)

        self.lidar_sample2 = data.ranges

        self.episode += 0.5 

        assert self.naction2 == self.steps

        # Updates number of lidar readings from robot 1
        self.naction2 += 1

        assert self.naction2 == self.steps + 1

    def get_point_cloud(self):
        point_cloud = []
        angle = math.pi / 2

        for dist in self.lidar_sample:
            point_cloud.append((self.position[0] + math.cos(angle) * dist, self.position[1] + math.sin(angle) * dist))
            angle += 0.06981317007

        assert 3*math.pi/2 - 0.07 <= angle <= 3*math.pi/2 + 0.07

        return point_cloud
    
    def get_point_cloud_clusters(self, tolerance):
        clusters = []
        curr_cluster = []

        for i in range(len(self.point_cloud) - 1):
            p1 = self.point_cloud[i]
            p2 = self.point_cloud[i+1]
            tolerance = self.lidar_sample[i] * (math.sin(math.pi / 46) / math.sin(0.3 - math.pi / 46)) + 0.045

            curr_cluster.append(p1)

            if self.euclid_dist(p1[0], p1[1], p2[0], p2[1]) > tolerance:
                clusters.append(curr_cluster)
                curr_cluster = []

        curr_cluster.append(self.point_cloud[-1])
        clusters.append(curr_cluster)

        return clusters

    def min_dist_robot(self, tolerance):
        ray_offset = 0

        for cluster in self.clusters:
            if len(cluster) == 1:
                ray_offset += 1
                continue

            cluster_length = self.euclid_dist(cluster[0][0], cluster[0][1], cluster[-1][0], cluster[-1][1])

            # 0.165 is the hardcoded length of the car model
            if 0.165 - tolerance <= cluster_length <= 0.165 + tolerance:
                dists = self.lidar_sample[ray_offset:ray_offset + len(cluster)]
                return (min(dists), (ray_offset + dists.index(min(dists))) * math.pi / 46)

            ray_offset += len(cluster)

        if (self.last_dist_diff > 0):
            return (10.0, math.pi)
        return (-10.0, 0)


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
        self.change_robot_position('robot1', 12.35, 5.0, 1.57079632679)
        self.change_robot_position('robot2', 12.35, 4.7, 1.57079632679)
        self.change_robot_position('prox', 12.35, 5.3, 0)

    def change_robot_position(self, name, x, y, theta):
        pose = Pose2D()
        pose.x = x
        pose.y = y
        pose.theta = theta
        self.move_model_service(name = name, pose = pose)

    def reset_counters(self):
        self.total_episodes = 0
        self.success_episodes = 0
        self.adjacent_episodes = 0
        self.total_steps = 0
        self.forward_steps = 0
        self.adj_steps = 0
        self.adjacencies = []

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


