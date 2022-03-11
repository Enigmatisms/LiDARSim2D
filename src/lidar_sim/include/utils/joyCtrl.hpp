/// @author (Enigmatisms:https://github.com/Enigmatisms) @copyright Qianyue He
#pragma once
#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Joy.h>
#include <atomic>
#include <thread>

class JoyCtrl {
public:
    JoyCtrl();
    ~JoyCtrl() {
        spinner->stop();
        delete spinner;
    };
public:
    void joyStickCallBack(const sensor_msgs::Joy::ConstPtr& msg);

    void start(ros::NodeHandle& nh);

    // Needs mapping from float to int32
    std::atomic_int32_t forward_speed;
    std::atomic_int32_t lateral_speed;
    std::atomic_int32_t angular_vel;

    // needs pos-edge detection
    std::atomic_bool record_bag;

    // needs neg-edge detection
    std::atomic_bool exit_flag;
private:
    ros::Subscriber joy_sub;
    ros::AsyncSpinner* spinner;
};