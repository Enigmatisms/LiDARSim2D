#include <ros/ros.h>
#include <array>
#include "mapEdit.h"
#include "consts.h"

std::vector<std::vector<cv::Point>> obsts;
std::vector<cv::Point> obstacle;
cv::Point start_point(0, 0);
cv::Mat src;
bool start_set = false;

void on_mouse(int event, int x,int y, int flags, void *ustc) {
    if (event == cv::EVENT_LBUTTONUP) {
        obstacle.emplace_back(x, y);
        printf("(%d, %d) pushed in.\n", x, y);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "editor");
    ros::NodeHandle nh;
    const cv::Rect rect(0, 0, 1200, 900);
    cv::Mat src(cv::Size(1200, 900), CV_8UC3);
    cv::rectangle(src, rect, cv::Scalar(100, 100, 100), -1);
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    std::string name = nh.param<std::string>("/editor/map_name", "test");
    std::string pack_path = getPackagePath();
    while (true) {
        mapDraw(obsts, obstacle, src);
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
        }
    }
    return 0;
}