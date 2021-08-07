#include "consts.h"

const std::array<cv::Point, 11> north_wall = {   
    cv::Point(30, 30), cv::Point(144, 30), cv::Point(258, 30), cv::Point(372, 30),
    cv::Point(486, 30), cv::Point(600, 30), cv::Point(714, 30), cv::Point(828, 30),
    cv::Point(942, 30), cv::Point(1056, 30), cv::Point(1170, 30)
};

const std::array<cv::Point, 9> east_wall = {
    cv::Point(1170, 30), cv::Point(1170, 135), cv::Point(1170, 240),
    cv::Point(1170, 345), cv::Point(1170, 450), cv::Point(1170, 555),
    cv::Point(1170, 660), cv::Point(1170, 765), cv::Point(1170, 870)
};       // 105

const std::array<cv::Point, 11> south_wall = {
    cv::Point(1170, 870), cv::Point(1056, 870), cv::Point(942, 870), cv::Point(828, 870),
    cv::Point(714, 870), cv::Point(600, 870), cv::Point(486, 870), cv::Point(372, 870),
    cv::Point(258, 870), cv::Point(144, 870), cv::Point(30, 870)
};      // 114

const std::array<cv::Point, 9> west_wall = {
    cv::Point(30, 870), cv::Point(30, 765), cv::Point(30, 660),
    cv::Point(30, 555), cv::Point(30, 450), cv::Point(30, 345),
    cv::Point(30, 240), cv::Point(30, 135), cv::Point(30, 30)
};
    