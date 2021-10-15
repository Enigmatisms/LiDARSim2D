#include "utils/consts.h"

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
// const std::array<cv::Point, 11> north_wall = {   
//     cv::Point(-1200, -1200), cv::Point(144, -1200), cv::Point(258, -1200), cv::Point(372, -1200),
//     cv::Point(486, -1200), cv::Point(600, -1200), cv::Point(714, -1200), cv::Point(828, -1200),
//     cv::Point(942, -1200), cv::Point(1056, -1200), cv::Point(2400, -1200)
// };

// const std::array<cv::Point, 9> east_wall = {
//     cv::Point(2400, -1200), cv::Point(2400, 135), cv::Point(2400, 240),
//     cv::Point(2400, 345), cv::Point(2400, 450), cv::Point(2400, 555),
//     cv::Point(2400, 660), cv::Point(2400, 765), cv::Point(2400, 2100)
// };       // 105

// const std::array<cv::Point, 11> south_wall = {
//     cv::Point(2400, 2100), cv::Point(1056, 2100), cv::Point(942, 2100), cv::Point(828, 2100),
//     cv::Point(714, 2100), cv::Point(600, 2100), cv::Point(486, 2100), cv::Point(372, 2100),
//     cv::Point(258, 2100), cv::Point(144, 2100), cv::Point(-1200, 2100)
// };      // 114

// const std::array<cv::Point, 9> west_wall = {
//     cv::Point(-1200, 2100), cv::Point(-1200, 765), cv::Point(-1200, 660),
//     cv::Point(-1200, 555), cv::Point(-1200, 450), cv::Point(-1200, 345),
//     cv::Point(-1200, 240), cv::Point(-1200, 135), cv::Point(-1200, -1200)
// };

std::string getPackagePath() {
    char string[256] = "rospack find lidar_sim";
    FILE* fp = popen(string, "r");
    char* res = fgets(string, 256, fp);
    pclose(fp);
    if (res == nullptr)
        return std::string();
    std::string path(string);
    path.pop_back();
    return path;
}
    