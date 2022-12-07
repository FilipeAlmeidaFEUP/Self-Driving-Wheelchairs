#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <flatland_msgs/Collisions.h>
#include <ros/console.h>
#include <stdio.h>

float speed = 1; 
int cd = 10;

class CustomRobotController
{
    public:
        CustomRobotController();

    private:
        void processLaserScan(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processProxScan(const sensor_msgs::LaserScan &lidar_scan_msg);
        void processCollision(const flatland_msgs::Collisions &collision_msg);
        ros::NodeHandle nh;
        std::string scan_topic, prox_topic, bumper_topic, twist_topic;
        ros::Subscriber scan_sub;
        ros::Subscriber bumper_sub;
        ros::Subscriber prox_sub;
        ros::Publisher twist_pub;

};

CustomRobotController::CustomRobotController():
    //scan_topic("/simple_lidar"), //uncomment only one of these lines at a time, depending on which LiDAR you wish to use
    scan_topic("/static_laser"), //"/static_laser" uses the complete LiDAR scan, "/simple_lidar" uses a simplified version
    prox_topic("/prox_laser"),
    bumper_topic("/collisions"),
    twist_topic("/cmd_vel")
{
    scan_sub = nh.subscribe(scan_topic, 1, &CustomRobotController::processLaserScan, this);
    prox_sub = nh.subscribe(prox_topic, 1, &CustomRobotController::processProxScan, this);
    bumper_sub = nh.subscribe(bumper_topic, 1, &CustomRobotController::processCollision, this);
    twist_pub = nh.advertise<geometry_msgs::Twist>(twist_topic, 1);
}

void CustomRobotController::processLaserScan(const sensor_msgs::LaserScan &lidar_scan_msg)
{
    geometry_msgs::Twist twist_msg;

    /**
        Write your code here

        The values from the LiDAR are in lidar_scan_msg.ranges
        For more information on the data contained in a LaserScan message, consult http://docs.ros.org/en/api/sensor_msgs/html/msg/LaserScan.html

        The following are used to set the linear and angular velocities of the robot
        twist_msg.linear.x = 0;
        twist_msg.angular.z = 0;
     */
    
    // Example 1: A robot that moves forward
    twist_msg.linear.x = speed; // A positive lenear velocity makes the robot move forward
    twist_msg.angular.z = 0; // Zero angular velocitiy prevents the robot from turning/rotating
    
    
    /**
    // Example 2: A robot that moves in a circle
    twist_msg.linear.x = 1; // A positive lenear velocity makes the robot move forward
    twist_msg.angular.z = -1; // A negative angular velocitiy makes the robot turn clockwise
    */
    
    // Example 3: A robot that moves in circles, and turns around itself when it finds an obstacle in front of it
    // Find if the laser in front of the robot is detecting an obstacle too close to them
    /*int num_lasers = lidar_scan_msg.ranges.size();
    const float max_distance = 1.00; // maximum allowed distance for an obstacle in one of three sensor readings before the robot starts spining around itself
    float middle_laser_range = lidar_scan_msg.ranges[num_lasers / 2];
    if(!(middle_laser_range >= lidar_scan_msg.range_min && middle_laser_range <= lidar_scan_msg.range_max) // if the distance reading is invalid or...
            || middle_laser_range < max_distance // ... the distance is below the maximum allowed distance
    ) {
        // ROTATE
        twist_msg.linear.x = 0;
        twist_msg.angular.z = -2;
    }
    else {
        // MOVE FORWARD
        twist_msg.linear.x = 2;
        twist_msg.angular.z = -2;
    }/**/
    

    twist_pub.publish(twist_msg);
}

void CustomRobotController::processProxScan(const sensor_msgs::LaserScan &lidar_scan_msg) {
    for (int i = 0; i < lidar_scan_msg.ranges.size(); i++) {
        if (lidar_scan_msg.ranges[i] < 0.25 && lidar_scan_msg.ranges[i] > 0 && cd == 10) {
            speed *= -1;
            cd--;
            break;
        }
    } 
    //ROS_INFO("dsa");
    if (cd != 10) cd--;
    if (cd == 0) cd = 10;
}

void CustomRobotController::processCollision(const flatland_msgs::Collisions &collision_msg) {
    if (collision_msg.collisions.size() > 0 && cd == 10) {
        speed *= -1;
        cd--;
    }
    if (cd != 10) cd--;
    if (cd == 0) cd = 10;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "wheelchair_controller");
  CustomRobotController robot_controller;

  ros::spin();
  
  return(0);
}
