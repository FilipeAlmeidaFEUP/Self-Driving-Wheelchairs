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
        void processLaserScan(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processProxScan(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processCollision(const flatland_msgs::Collisions &collision_msg);
        void resetPosition();
        ros::NodeHandle nh;
        std::string scan_topic, prox_topic, bumper_topic, twist_topic;
        ros::Subscriber scan_sub;
        ros::Subscriber bumper_sub;
        ros::Subscriber prox_sub;
        ros::Publisher twist_pub;
        ros::ServiceClient client;
};

CustomRobotController::CustomRobotController():
    //scan_topic("/simple_lidar"), //uncomment only one of these lines at a time, depending on which LiDAR you wish to use
    scan_topic("/static_laser1"), //"/static_laser" uses the complete LiDAR scan, "/simple_lidar" uses a simplified version
    prox_topic("/prox_laser1"),
    bumper_topic("/collisions"),
    twist_topic("/cmd_vel1")
{
    scan_sub = nh.subscribe(scan_topic, 1, &CustomRobotController::processLaserScan, this);
    prox_sub = nh.subscribe(prox_topic, 1, &CustomRobotController::processProxScan, this);
    bumper_sub = nh.subscribe(bumper_topic, 1, &CustomRobotController::processCollision, this);
    twist_pub = nh.advertise<geometry_msgs::Twist>(twist_topic, 1);

    client = nh.serviceClient<flatland_msgs::MoveModel>("/move_model");
}

void CustomRobotController::processLaserScan(const sensor_msgs::LaserScan &lidar_scan_msg)
{

    //ROS_INFO("%d", rand() % 7);
    geometry_msgs::Twist twist_msg;
    
    int action = rand() % 7;
    twist_msg.linear.x = actions[action][0]; 
    twist_msg.angular.z = actions[action][1]; 
    

    twist_pub.publish(twist_msg);
}

void CustomRobotController::processProxScan(const sensor_msgs::LaserScan &lidar_scan_msg) {
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
    twist_pub.publish(twist_msg);

    flatland_msgs::MoveModel srv_msg;
    srv_msg.request.name = "robot1";
    srv_msg.request.pose.x = 0.6;
    srv_msg.request.pose.y = 0.5;
    srv_msg.request.pose.theta = 1.57079632679;
    client.call(srv_msg);
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "wheelchair_controller1");
  CustomRobotController robot_controller;

  ros::spin();
  
  return(0);
}
