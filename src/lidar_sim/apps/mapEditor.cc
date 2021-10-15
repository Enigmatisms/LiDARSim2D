/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#include <ros/ros.h>
#include <array>
#include "utils/mapEdit.h"
#include "utils/consts.h"

constexpr double deg2rad = M_PI / 180.0; 
double angle_incre = 0.0;
const cv::Rect walls(0, 0, 1200, 900);
const cv::Rect floors(30, 30, 1140, 840);
const std::array<cv::Scalar, 3> colors = {
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(0, 0, 255)
};
std::array<cv::Point, 2> tasks = {cv::Point(-1, -1), cv::Point(-1, -1)};
std::vector<std::vector<cv::Point>> obsts;
std::vector<cv::Point> obstacle;
cv::Point start_point(0, 0);
cv::Mat src;
bool start_set = false, refresh_flag = false;
// bool straight_line_mode = false;
// bool rectangle_mode = false;
// bool circle_mode = false;
std::array<bool, 3> status = {false, false};
int last_set = -1;
int CIRCLE_RESOLUTION = 300;

void reset() {
    start_set = false;
    tasks[0] = cv::Point(-1, -1);
    tasks[1] = cv::Point(-1, -1);
}

void statusSet(int id) {
    status[id] = !status[id];
    last_set = -1;
    if (status[id] == true) {
        for (int i = 0; i < 3; i++) {
            if (i == id) continue;
            status[i] = false;
        }
        last_set = id;
    }
    reset();
    refresh_flag = true;
}

void on_mouse(int event, int x,int y, int flags, void *ustc) {
    if (event == cv::EVENT_MOUSEMOVE) {
        if (status[0] == true && start_set == true) {
            refresh_flag = true;
            const cv::Point& last_point = obstacle.back();
            const cv::Point diff = cv::Point(x, y) - last_point;
            if (std::abs(diff.x) > std::abs(diff.y))
                y = last_point.y;
            else x = last_point.x;
            tasks[1] = cv::Point(x, y);
        } else if ((status[1] == true || status[2] == true) && start_set == true) {
            refresh_flag = true;
            tasks[1] = cv::Point(x, y);
        }
    }
    if (event == cv::EVENT_LBUTTONUP) {
        bool emplace_flag = true;
        if (status[0] == true && start_set) {
            const cv::Point& last_point = obstacle.back();
            const cv::Point diff = cv::Point(x, y) - last_point;
            if (std::abs(diff.x) > std::abs(diff.y))
                y = last_point.y;
            else x = last_point.x;
            tasks[0] = cv::Point(x, y);
        } else if (status[1] == true && start_set) {
            emplace_flag = false;
            const cv::Point diff = cv::Point(x, y) - start_point;
            obstacle.push_back(start_point);
            if ((diff.x > 0 && diff.y > 0) || (diff.x < 0 && diff.y < 0)) {
                obstacle.emplace_back(start_point.x, y);
                obstacle.emplace_back(x, y);
                obstacle.emplace_back(x, start_point.y);
            } else {
                obstacle.emplace_back(x, start_point.y);
                obstacle.emplace_back(x, y);
                obstacle.emplace_back(start_point.x, y);
            }
            obsts.emplace_back(obstacle);
            obstacle.clear();
            printf("One obstacle is added.\n");
            reset();
            return;
        } else if (status[2] == true && start_set) {
            emplace_flag = false;
            const cv::Point diff = cv::Point(x, y) - start_point;
            const double radius = std::sqrt(diff.dot(diff));
            double start_angle = atan2(diff.y, diff.x);
            obstacle.emplace_back(x, y);
            for (int i = 1; i < CIRCLE_RESOLUTION; i++) {
                start_angle += angle_incre;
                obstacle.emplace_back(
                    start_point.x + radius * cos(start_angle),
                    start_point.y + radius * sin(start_angle)
                );
            }
            statusSet(2);
            return;
        }
        if (last_set >= 0) {
            if (start_set == false) {
                start_set = true;
                start_point = cv::Point(x, y);
            }
            if (last_set > 0)
                emplace_flag = false;
            tasks[0] = cv::Point(x, y);
        }
        if (emplace_flag == true)
            obstacle.emplace_back(x, y);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "editor");
    ros::NodeHandle nh;
    const cv::Rect rect(0, 0, 1200, 900);
    cv::Mat src(cv::Size(1200, 900), CV_8UC3);
    cv::rectangle(src, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, floors, cv::Scalar(100, 100, 100), -1);
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    std::string name = nh.param<std::string>("/editor/map_name", "test");
    CIRCLE_RESOLUTION = nh.param<int>("/editor/circle_reso", 300);
    std::string pack_path = getPackagePath();
    angle_incre = (300.0 / static_cast<double>(CIRCLE_RESOLUTION)) * deg2rad;
    while (true) {
        if (refresh_flag == true) {
            refresh_flag = false;
            cv::rectangle(src, walls, cv::Scalar(10, 10, 10), -1);
            cv::rectangle(src, floors, cv::Scalar(100, 100, 100), -1);
            for (int i = 0; i < 3; i++) {
                if (status[i]) 
                    cv::circle(src, cv::Point(15 + i * 30, 15), 10, colors[i], -1);
            }
        }
        mapDraw(obsts, obstacle, src);
        extraTask(tasks, src, last_set);
        cv::imshow("disp", src);
        char key = cv::waitKey(10);
        if (key == 27) {
            printf("Exiting directly, map not saved.\n");
            break;
        } else if (key == 'e') {
            if (obstacle.size() < 3) continue;
            obsts.emplace_back(obstacle);
            obstacle.clear();
            printf("One obstacle is added.\n");
        } else if (key == 's') {
            if (obstacle.size() >= 3) {
                obsts.emplace_back(obstacle);
                obstacle.clear();
            }
            mapSave(obsts, pack_path + "/../../maps/" + name + ".txt");
            printf("Map saved.\n");
            break;
        } else if (key == 'p') {
            if (obstacle.size() > 0) {
                obstacle.pop_back();
                refresh_flag = true;
            } else {
                if (obsts.size() > 0) {
                    obstacle.assign(obsts.back().begin(), obsts.back().end());
                    obsts.pop_back();
                    refresh_flag = true;
                } else {
                    printf("Nothing to pop.\n");
                }
            }
        } else if (key == 'l') {
            statusSet(0);
        } else if (key == 'r') {
            statusSet(1);
        } else if (key == 'c') {
            statusSet(2);
        }
    }
    return 0;
}