#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <chrono>
#include <thread>
#include <array>
#include "mapEdit.h"
#include "consts.h"
#include "cuda_pf.hpp"

cv::Mat src;
Eigen::Vector2d obs;
int x_motion = 0, y_motion = 0;
double angle = 0.0, delta_angle = 0.0;
bool obs_set = false;

void on_mouse(int event, int x, int y, int flags, void *ustc) {
    if (event == cv::EVENT_LBUTTONDOWN && obs_set == false) {
        printf("cv::Point(%d, %d),\n", x, y);
        obs(0) = double(x);
        obs(1) = double(y);
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
    int resample_freq = nh.param<int>("/cuda_test/resample_freq", 3);
    int point_num = nh.param<int>("/cuda_test/point_num", 27000);
    double trans_vel = nh.param<double>("/cuda_test/trans_vel", 3);
    double rot_vel = nh.param<double>("/cuda_test/rot_vel", 1.0);
    double angle_min = nh.param<double>("/cuda_test/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/cuda_test/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/cuda_test/angle_incre", M_PI / 360);
    obs_set = nh.param<bool>("/cuda_test/resquire_no_init_pos", true);
    std::string dev_name = nh.param<std::string>("/cuda_test/dev_name", "/dev/input/by-id/usb-Keychron_Keychron_K2-event-kbd");

    rot_vel = rot_vel * M_PI / 180.0;

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
    cv::threshold(occupancy, occupancy, 20, 255, cv::THRESH_BINARY_INV);
    cv::dilate(occupancy, occupancy, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    for (const Obstacle& egs: obstacles) {
        cv::circle(src, egs.front(), 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(src, egs.back(), 3, cv::Scalar(255, 0, 0), -1);
    }    
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    obs = Eigen::Vector2d(660, 94);
    while (obs_set == false) {
        cv::imshow("disp", src);
        char key = cv::waitKey(10);
        if (key == 27)
            return 0;
    }
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    CudaPF cpf(occupancy, angles, point_num, resample_freq);
    cpf.intialize(obstacles);
    cpf.particleInitialize(occupancy, Eigen::Vector3d(obs.x(), obs.y(), 0.0));
    bool render_flag = true;
    double time_cnt = 1.0, time_sum = 0.0;
    double start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    double end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    Eigen::Vector2d translation = Eigen::Vector2d::Zero();
    time_sum += end_t - start_t;
    printf("Main started.\n");
    while (true) {
        cv::imshow("disp", src);
        char key = cv::waitKey(1);
        bool break_flag = false;
        if (render_flag == true) {
            start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            Eigen::Vector3d act_obs;
            act_obs << obs, angle;            
            cpf.particleUpdate(x_motion, y_motion, delta_angle);
            cpf.filtering(obstacles, act_obs, src);
            // cpf.singleDebugDemo(obstacles, act_obs, src);
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
                translation(0) = cos(angle) * trans_vel;
                translation(1) = sin(angle) * trans_vel;
                render_flag = true;
                break;
            }
            case 'a': {
                translation(0) = sin(angle) * trans_vel;
                translation(1) = -cos(angle) * trans_vel;
                render_flag = true;
                break;
            }
            case 's': {
                translation(0) = -cos(angle) * trans_vel;
                translation(1) = -sin(angle) * trans_vel;
                render_flag = true;
                break;
            }
            case 'd': {
                translation(0) = -sin(angle) * trans_vel;
                translation(1) = cos(angle) * trans_vel;
                render_flag = true;
                break;
            }
            case 'p': {
                angle += rot_vel;
                delta_angle = rot_vel;
                if (angle > M_PI)
                    angle -= 2 * M_PI;
                render_flag = true;
                break;
            }
            case 'o': {
                angle -= rot_vel;
                delta_angle = -rot_vel;
                if (angle < -M_PI)
                    angle += 2 * M_PI;
                render_flag = true;
                break;
            }
            case 27: break_flag = true; break;
        }
        if (break_flag)
            break;
        if (render_flag == false) continue; 
        Eigen::Vector2d tmp = obs + translation;
        if (occupancy.at<uchar>(int(tmp.y()), int(tmp.x())) > 0) {
            if (occupancy.at<uchar>(int(tmp.y()), int(obs.x())))
                translation(1) = 0.0;
            if (occupancy.at<uchar>(int(obs.y()), int(tmp.x())))
                translation(0) = 0.0;
        }
        obs += translation;
        x_motion = translation(0);
        y_motion = translation(1);
        translation.setZero();
    }
    double mean_time = time_sum / time_cnt;
    printf("Average running time: %.6lf ms, fps: %.6lf hz\n", mean_time * 1e3, 1.0 / mean_time);
    cv::destroyAllWindows();
    return 0;
}

