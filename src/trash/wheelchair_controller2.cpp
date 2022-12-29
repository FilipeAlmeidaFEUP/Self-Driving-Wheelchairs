#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/LaserScan.h>
#include <flatland_msgs/Collisions.h>
#include <flatland_msgs/MoveModel.h>
#include <ros/console.h>
#include <stdio.h>
#include <time.h>

float linear_speed = 0.5;
float angular_speed = 1;

float actions[7][2] = {
    {0, 0},
    {linear_speed, 0},
    {-linear_speed, 0},
    {0, angular_speed},
    {0, -angular_speed},
    {linear_speed, angular_speed},
    {linear_speed, -angular_speed},
};

class CustomRobotController
{
    public:
        CustomRobotController();

    private:
        void processLaserScan1(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processLaserScan2(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processProxScan1(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processProxScan2(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processCollision(const flatland_msgs::Collisions &collision_msg);
        void resetPosition();
        ros::NodeHandle nh;
        std::string scan_topic1, scan_topic2, prox_topic1, prox_topic2, bumper_topic, twist_topic1, twist_topic2;
        ros::Subscriber scan_sub1;
        ros::Subscriber scan_sub2;
        ros::Subscriber bumper_sub;
        ros::Subscriber prox_sub1;
        ros::Subscriber prox_sub2;
        ros::Publisher twist_pub1;
        ros::Publisher twist_pub2;
        ros::ServiceClient client;
};

CustomRobotController::CustomRobotController():
    scan_topic1("/static_laser1"), 
    scan_topic2("/static_laser2"), 
    prox_topic1("/prox_laser1"),
    prox_topic2("/prox_laser2"),
    bumper_topic("/collisions"),
    twist_topic1("/cmd_vel1"),
    twist_topic2("/cmd_vel2")
{
    scan_sub1 = nh.subscribe(scan_topic1, 1, &CustomRobotController::processLaserScan1, this);
    scan_sub2 = nh.subscribe(scan_topic2, 1, &CustomRobotController::processLaserScan2, this);
    prox_sub1 = nh.subscribe(prox_topic1, 1, &CustomRobotController::processProxScan1, this);
    prox_sub2 = nh.subscribe(prox_topic2, 1, &CustomRobotController::processProxScan2, this);
    bumper_sub = nh.subscribe(bumper_topic, 1, &CustomRobotController::processCollision, this);
    twist_pub1 = nh.advertise<geometry_msgs::Twist>(twist_topic1, 1);
    twist_pub2 = nh.advertise<geometry_msgs::Twist>(twist_topic2, 1);

    client = nh.serviceClient<flatland_msgs::MoveModel>("/move_model");
}

void CustomRobotController::processLaserScan1(const sensor_msgs::LaserScan &lidar_scan_msg)
{

    //ROS_INFO("%d", rand() % 7);
    geometry_msgs::Twist twist_msg;
    
    int action = rand() % 7;
    twist_msg.linear.x = actions[action][0]; 
    twist_msg.angular.z = actions[action][1]; 
    

    twist_pub1.publish(twist_msg);
}

void CustomRobotController::processLaserScan2(const sensor_msgs::LaserScan &lidar_scan_msg)
{

    //ROS_INFO("%d", rand() % 7);
    geometry_msgs::Twist twist_msg;
    
    int action = rand() % 7;
    twist_msg.linear.x = actions[action][0]; 
    twist_msg.angular.z = actions[action][1]; 
    

    twist_pub2.publish(twist_msg);
}

void CustomRobotController::processProxScan1(const sensor_msgs::LaserScan &lidar_scan_msg) {
    for (int i = 0; i < lidar_scan_msg.ranges.size(); i++) {
        if (lidar_scan_msg.ranges[i] < 0.25 && lidar_scan_msg.ranges[i] > 0) {
            this->resetPosition();
            break;
        }
    } 
}

void CustomRobotController::processProxScan2(const sensor_msgs::LaserScan &lidar_scan_msg) {
    for (int i = 0; i < lidar_scan_msg.ranges.size(); i++) {
        if (lidar_scan_msg.ranges[i] < 0.25 && lidar_scan_msg.ranges[i] > 0) {
            this->resetPosition();
            break;
        }
    } 
}

void CustomRobotController::processCollision(const flatland_msgs::Collisions &collision_msg) {
    if (collision_msg.collisions.size() > 0) {
        this->resetPosition();
    }
    
}

void CustomRobotController::resetPosition() {
    geometry_msgs::Twist twist_msg;
    twist_msg.linear.x = 0; 
    twist_msg.angular.z = 0;
    twist_pub1.publish(twist_msg);
    twist_pub2.publish(twist_msg);

    flatland_msgs::MoveModel srv_msg;
    srv_msg.request.name = "robot1";
    srv_msg.request.pose.x = 0.4;
    srv_msg.request.pose.y = 0.5;
    srv_msg.request.pose.theta = 1.57079632679;
    client.call(srv_msg);

    srv_msg.request.name = "robot2";
    srv_msg.request.pose.x = 0.8;
    client.call(srv_msg);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "wheelchair_controller2");
  CustomRobotController robot_controller;

  ros::spin();
  
  return(0);
}
