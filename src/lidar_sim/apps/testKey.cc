/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#include <ros/ros.h>
#include <thread>
#include <bitset>
#include "utils/keyCtrl.hpp"

const std::string dev_name = "/dev/input/by-path/platform-i8042-serio-0-event-kbd";

int main(int argc, char** argv) {
    ros::init(argc, argv, "key_test");
    KeyCtrl kc(dev_name);
    std::thread th(&KeyCtrl::onKeyThread, &kc);
    th.detach();
    char last_val = status;
    while (true) {
        if (status != 0x00) {
            last_val = status;
            std::bitset<8> x(last_val);
            std::cout << x << std::endl;
            usleep(10000);
        } else {
            if (last_val > 0) {
                last_val = 0x00;
                std::bitset<8> x(0x00);
                std::cout << x << std::endl;
            }
        }
    }
    return 0;
}