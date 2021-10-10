#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <chrono>
#include "mapEdit.h"
#include "consts.h"
#include "cuda_pf.hpp"

cv::Mat src;
cv::Point obs;
int x_motion = 0, y_motion = 0;
double angle = 0.0, delta_angle = 0.0;
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
    ros::init(argc, argv, "cuda_test");
    ros::NodeHandle nh;
    cv::setNumThreads(4);
    std::vector<std::vector<cv::Point>> obstacles;

    std::string name = nh.param<std::string>("/cuda_test/map_name", "standard");
    int speed = nh.param<int>("/cuda_test/speed", 3);
    double rot_vel = nh.param<double>("/cuda_test/rot_vel", 1.0);
    double angle_min = nh.param<double>("/cuda_test/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/cuda_test/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/cuda_test/angle_incre", M_PI / 360);

    std::string pack_path = getPackagePath();
    printf("Package prefix: %s\n", pack_path.c_str());
    mapLoad(pack_path + "/../../maps/" + name + ".txt", obstacles);
    printf("Map loaded.\n");
    src.create(cv::Size(1200, 900), CV_8UC3);
    cv::rectangle(src, cv::Rect(0, 0, 1200, 900), cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, cv::Rect(30, 30, 1140, 840), cv::Scalar(40, 40, 40), -1);
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
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    std::cout << "Here1\n";
    CudaPF cpf(occupancy, angles, 64000);
    std::cout << "Here2\n";

    cpf.intialize(obstacles);
    std::cout << "Here3\n";

    cpf.particleInitialize(occupancy);
    std::cout << "Here4\n";
    bool render_flag = true;
    double time_cnt = 1.0, time_sum = 0.0;
    double start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    double end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    time_sum += end_t - start_t;
    printf("Main started.\n");
    while (true) {
        cv::imshow("disp", src);
        char key = cv::waitKey(1);
        bool break_flag = false;
        if (render_flag == true) {
            start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            Eigen::Vector3d act_obs(obs.x, obs.y, angle);
            cpf.particleUpdate(x_motion, y_motion, delta_angle);
            cpf.filtering(obstacles, act_obs, src);
            cpf.singleDebugDemo(obstacles, act_obs, src);
            end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            x_motion = 0;
            y_motion = 0;
            // outputVideo.write(src);
            time_sum += end_t - start_t;
            time_cnt += 1.0;
            render_flag = false;
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
            case 'o': {
                angle -= rot_vel;
                delta_angle = -rot_vel;
                if (angle < -M_PI)
                    angle += 2 * M_PI;
            }
            case 'p': {
                angle += rot_vel;
                delta_angle = +rot_vel;
                if (angle > M_PI)
                    angle -= 2 * M_PI;
            }
            case 27: break_flag = true;
        }
        if (break_flag == true)
            break;
    }
    double mean_time = time_sum / time_cnt;
    printf("Average running time: %.6lf ms, fps: %.6lf hz\n", mean_time * 1e3, 1.0 / mean_time);
    cv::destroyAllWindows();
    return 0;
}

