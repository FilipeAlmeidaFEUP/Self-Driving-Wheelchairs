#!/usr/bin/env python3

import time
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel
from nav_msgs.msg import Odometry 
import numpy as np
from tf.transformations import euler_from_quaternion
import math


def distance_handler(direction, dist_values):
    maxSpeed: float = 0.1
    distP: float = 7.5
    angleP: float = 5.75
    wallDist: float = 0.25

    # Find the angle of the ray that returned the minimum distance
    size: int = len(dist_values)
    min_index: int = 0
    if direction == -1:
        min_index = size - 1
    for i in range(size):
        idx: int = i
        if direction == -1:
            idx = size - 1 - i
        if dist_values[idx] < dist_values[min_index] and dist_values[idx] > 0.0:
            min_index = idx

    angle_increment: float = 2*math.pi / (size - 1)
    angleMin: float = (size // 2 - min_index) * angle_increment
    distMin: float = dist_values[min_index]
    distFront: float = dist_values[size // 2]
    distSide: float = dist_values[size // 4] if (direction == 1) else dist_values[3*size // 4]
    distBack: float = dist_values[0]

    # Prepare message for the robot's motors
    linear_vel: float
    angular_vel: float


    # Decide the robot's behavior
    if math.isfinite(distMin):
        if distFront < 1.25*wallDist and (distSide < 1.25*wallDist or distBack < 1.25*wallDist):
            # UNBLOCK
            angular_vel = direction * -1
        else:
            # REGULAR
            angular_vel = direction * distP * (distMin - wallDist) + angleP * (angleMin - direction * math.pi / 2)
        if distFront < wallDist:
            # TURN
            linear_vel = 0
        elif distFront < 2 * wallDist or distMin < wallDist * 0.75 or distMin > wallDist * 1.25:
            # SLOW
            linear_vel = 0.5 * maxSpeed
        else:
            # CRUISE
            linear_vel = maxSpeed
    else:
        # WANDER
        angular_vel = np.random.normal(loc=0.0, scale=1.0)
        linear_vel = maxSpeed

    return linear_vel, angular_vel

def change_robot_speed(linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        twist_topic.publish(twist_msg)

def sample_lidar(data):
        global test
        if test:
            print(data.ranges[:math.floor(len(data.ranges) / 2)])
            print(len(data.ranges[:math.floor(len(data.ranges) / 2)]))
            print(min(data.ranges[:math.floor(len(data.ranges) / 2)]))

        real_data = data.ranges[::-1]
        linear_vel, angular_vel = distance_handler(1, real_data)

        change_robot_speed(linear_vel, angular_vel)

def check_collisions(data):
        global collided
        if collided: return
        if len(data.collisions) > 0:
            change_robot_speed(0,0)
            collided = True

def check_finished(data):
        global finished
        if finished: return
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            if(min(values) < 0.25):
                change_robot_speed(0,0)
                finished = True

if __name__ == '__main__':
    global twist_topic, collided, finished, test
    test = False
    collided = False
    finished = False

    # Initialize rospy    
    rospy.init_node('baseline_wall_following', anonymous=True)

    # Initialize subscribers/services. See one wheelchair env
    rospy.Subscriber("/static_laser1" , LaserScan, sample_lidar, buff_size=10000000, queue_size=1)
    rospy.Subscriber("/collisions", Collisions, check_collisions, buff_size=10000000, queue_size=1)
    rospy.Subscriber("/prox_laser1", LaserScan, check_finished, buff_size=10000000, queue_size=1)
    twist_topic = rospy.Publisher("/cmd_vel1", Twist, queue_size=1)
    rospy.wait_for_service("/move_model")
    move_model_service = rospy.ServiceProxy("/move_model", MoveModel)

    # move bodies to correct coordinates
    chair_pose = Pose2D()
    goal_pose = Pose2D()

    chair_pose.x = 0.4
    chair_pose.y = 3.5
    chair_pose.theta = 1.57079632679
    goal_pose.x = 0.725
    goal_pose.y = 5.5
    goal_pose.theta = 1.57079632679

    move_model_service(name = 'robot1', pose = chair_pose)
    move_model_service(name = 'prox', pose = goal_pose)

    # TODO: Start counting time
    start_time = time.time()
    
    # TODO: Infinite loop while chair doesn't reach goal
    while (not (collided or finished)):
        pass

    if (collided):
        print("Failure!")
    
    if (finished):
        print("Cool!")

    # TODO: Stop timer, print time elapsed
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time}")
