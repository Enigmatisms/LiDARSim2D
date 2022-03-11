#include <chrono>
#include "utils/joyCtrl.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_joy");
    ros::NodeHandle nh;

    JoyCtrl jc;
    ros::Rate rate(50);
    jc.start(nh);
    while (ros::ok()) {
        printf("%lu\n", std::chrono::system_clock::now().time_since_epoch().count());
        rate.sleep();
    }
    ros::waitForShutdown();
    return 0;
}