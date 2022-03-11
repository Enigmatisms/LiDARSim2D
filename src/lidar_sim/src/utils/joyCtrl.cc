#include "utils/joyCtrl.hpp"

JoyCtrl::JoyCtrl() {
    spinner = new ros::AsyncSpinner(1);
    record_bag = false;
    exit_flag = false;
}

void JoyCtrl::start(ros::NodeHandle& nh) {
    joy_sub = nh.subscribe("joy", 8, &JoyCtrl::joyStickCallBack, this);
    if (spinner->canStart()) {
        spinner->start();
        printf("Spinner started.\n");
    } else {
        std::cerr << "Error: Spinner failed to start\n";
    }
}

void JoyCtrl::joyStickCallBack(const sensor_msgs::Joy::ConstPtr& msg) {
    float f_speed = msg->axes[1], l_speed = msg->axes[0], turn = msg->axes[3];
    forward_speed = reinterpret_cast<int &>(f_speed);
    lateral_speed = reinterpret_cast<int &>(l_speed);
    angular_vel = reinterpret_cast<int &>(turn);
    if (msg->buttons[4])
        record_bag = !record_bag;
    if (msg->buttons[5])
        exit_flag = true;
}