#include <iostream>
#include <unordered_map>
#include "utils/keyCtrl.hpp"

#define IS_USB_KBD 0
#if IS_USB_KBD == 0
    #define NULL_VALUE 57
#else
    #define NULL_VALUE ' ' 
#endif

std::atomic_char status(0x00);

const std::unordered_map<int, int> mapping = {
    {458778, 0}, {458756, 1}, {458774, 2}, {458759, 3}, {458773, 4}, {458770, 5}, {458771, 6}, {458793, 7},
    {17, 0}, {30, 1}, {31, 2}, {32, 3}, {19, 4}, {24, 5}, {25, 6}, {1, 7}
}; // w a s d r o p [esc]

KeyCtrl::KeyCtrl(std::string dev_name) {
    constexpr char name[13] = "K_MEDIUMRAW";
    const char *device = NULL;
    if ((getuid()) != 0) {          // admin clearance
        std::string command = "echo \"'\" | sudo -S chmod 777 " + dev_name;
        system(command.c_str());
    }
    device = dev_name.c_str();

    if ((fd = open(device, O_RDONLY)) == -1)                
        printf("%s is not a vaild device.\n", device);

    ioctl(fd, EVIOCGNAME(sizeof(name)), name);
    printf("Reading From : %s (%s), %d\n", device, name, fd);
}

KeyCtrl::~KeyCtrl() {
    close(fd);
}

void KeyCtrl::onKeyThread() {
    struct input_event ev[64];
    int rd = 0, value = 0, size = sizeof(input_event);
    while (true) {
        if ((rd = read(fd, ev, size * 64)) < size) {
            std::cerr << "Error occurred in key control.\n";
            perror_exit("read()"); 
        }
        value = ev[0].value;
        // printf("value: %d, %d\n", value, ev[1].value);
        if (value != NULL_VALUE && ev[1].value >= 1) {
            // key down    
            std::unordered_map<int, int>::const_iterator cit = mapping.find(value);
            if (cit == mapping.cend()) continue;
            int id = cit->second;
            char digit = (0x01 << id);
            status |= digit;
        } else {
            std::unordered_map<int, int>::const_iterator cit = mapping.find(value);
            if (cit == mapping.cend()) continue;
            int id = cit->second;
            char digit = ((0xfe) >> (8 - id) | (0xfe << id));
            status &= digit;
        }
    }
}