/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <chrono>
#include "utils/mapEdit.h"
#include "utils/consts.h"
#include "volume/ParticleFilter.hpp"

cv::Mat src;
cv::Point obs;
int x_motion = 0, y_motion = 0;
bool obs_set = false;

void on_mouse(int event, int x,int y, int flags, void *ustc) {
    if (event == cv::EVENT_LBUTTONDOWN && obs_set == false) {
        printf("cv::Point(%d, %d),\n", x, y);
        obs.x = x;
        obs.y = y;
        cv::circle(src, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        obs_set = true;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "filter");
    ros::NodeHandle nh;
    cv::setNumThreads(2);
    std::vector<std::vector<cv::Point>> obstacles;
    std::string name = nh.param<std::string>("/filter/map_name", "standard");
    int speed = nh.param<int>("/filter/speed", 3);
    std::string pack_path = getPackagePath();
    printf("Package prefix: %s\n", pack_path.c_str());
    mapLoad(pack_path + "/../../maps/" + name + ".txt", obstacles);
    printf("Map loaded.\n");
    src.create(cv::Size(1200, 900), CV_8UC3);
    cv::rectangle(src, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, floors, cv::Scalar(40, 40, 40), -1);
    cv::drawContours(src, obstacles, -1, cv::Scalar(10, 10, 10), -1);
    cv::Mat occupancy;
    cv::cvtColor(src, occupancy, cv::COLOR_BGR2GRAY);
    cv::threshold(occupancy, occupancy, 20, 255, cv::THRESH_BINARY);
    cv::erode(occupancy, occupancy, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    
    for (const Obstacle& egs: obstacles) {
        cv::circle(src, egs.front(), 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(src, egs.back(), 3, cv::Scalar(255, 0, 0), -1);
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    obs = cv::Point(367, 769);
    while (obs_set == false) {
        cv::imshow("disp", src);
        char key = cv::waitKey(10);
        if (key == 27)
            return 0;
    }
    ParticleFilter pf(occupancy, 2.0 * M_PI / 180.0, 12800);
    pf.particleInitialize(occupancy);
    bool render_flag = true;
    double time_cnt = 1.0, time_sum = 0.0;
    double start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    double end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    time_sum += end_t - start_t;
    // cv::imwrite("../asset/thumbnail2.png", src);
    // std::string outPath = "/home/sentinel/cv_output.avi";
    // cv::Size sWH = cv::Size(1200, 900);
	// cv::VideoWriter outputVideo;
	// outputVideo.open(outPath, 1482049860, 1.5, sWH);	    // DIVX
    printf("Main started.\n");
    // int img_cnt = 0;
    while (true) {
        cv::imshow("disp", src);
        char key = cv::waitKey(1);
        bool break_flag = false;
        if (render_flag == true) {
            start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            pf.particleUpdate(x_motion, y_motion);
            pf.filtering(obstacles, Eigen::Vector2d(obs.x, obs.y), src);
            end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            x_motion = 0;
            y_motion = 0;
            // outputVideo.write(src);
            time_sum += end_t - start_t;
            time_cnt += 1.0;
            render_flag = false;
            // std::string name = "../asset/img" + std::to_string(img_cnt++) + ".png";
            // cv::imwrite(name.c_str(), src);
        }
        switch(key) {
            case 'w': {
                if (obs.y > 30) {
                    obs.y -= speed;
                    y_motion -= speed;
                    render_flag = true;
                }
                break;
            }
            case 'a': {
                if (obs.x > 30) {
                    obs.x -= speed;
                    x_motion -= speed;
                    render_flag = true;
                }
                break;
            }
            case 's': {
                if (obs.y < 870) {
                    obs.y += speed;
                    y_motion += speed;
                    render_flag = true;
                }
                break;
            }
            case 'd': {
                if (obs.x < 1170) {
                    obs.x += speed;
                    x_motion += speed;
                    render_flag = true;
                }
                break;
            }
            case 27: break_flag = true;
        }
        if (break_flag == true)
            break;

    }
    double mean_time = time_sum / time_cnt;
    printf("Average running time: %.6lf ms, fps: %.6lf hz\n", mean_time * 1e3, 1.0 / mean_time);
    cv::destroyAllWindows();
    // outputVideo.release();
    return 0;
}
