/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <chrono>
#include <thread>
#include <array>
#include "utils/mapEdit.h"
#include "utils/consts.h"
#include "cuda/shadow_cast.hpp"

cv::Mat src;
Eigen::Vector2d obs;
int x_motion = 0, y_motion = 0;
double angle = -M_PI / 2., delta_angle = 0.0;
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

void from_cv_points(const std::vector<std::vector<cv::Point>>& pts, std::vector<Vec2>& outputs, std::vector<char>& next_ids) {
    for (const std::vector<cv::Point>&  pt_vec: pts) {
        std::vector<char> ids(pt_vec.size(), 0);
        ids.front() = static_cast<char>(pt_vec.size() - 1);
        ids.back() = -static_cast<char>(pt_vec.size() - 1);
        next_ids.reserve(next_ids.size() + ids.size());
        next_ids.insert(next_ids.end(), ids.begin(), ids.end());
        for (const cv::Point& pt: pt_vec) { 
            outputs.emplace_back(pt.x, pt.y);
        }
    }
    for (size_t i = 0; i < 4; i++) {
        outputs.emplace_back(reverse_wall[i].x, reverse_wall[i].y);
        next_ids.emplace_back(reverse_wall_ids[i]); 
    }
}

void draw_from_vec(const std::vector<std::vector<cv::Point>>& obstcs, const Eigen::Vector2d& obs, std::vector<Vec2>&& corners, cv::Mat& dst) {
    std::vector<std::vector<cv::Point>> results;
    results.emplace_back();
    for (const auto& elem: corners) {
        int x = min(max(0, int(elem.x)), 1199);
        int y = min(max(0, int(elem.y)), 899);
        results.back().emplace_back(x, y);
        // printf("(%d, %d), ", x, y);
    }
    // printf("\n");
    cv::rectangle(dst, cv::Rect(0, 0, 1200, 900), cv::Scalar(10, 10, 10), -1);
    cv::rectangle(dst, cv::Rect(30, 30, 1140, 840), cv::Scalar(40, 40, 40), -1);
    cv::drawContours(dst, obstcs, -1, cv::Scalar(10, 10, 10), -1);
    cv::drawContours(dst, results, -1, cv::Scalar(225, 225, 225), -1);
    cv::circle(dst, cv::Point(obs.x(), obs.y()), 4, cv::Scalar(0, 0, 255), -1);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "shadow_cast");
    ros::NodeHandle nh;
    cv::setNumThreads(4);
    std::vector<std::vector<cv::Point>> obstacles;

    std::string name = nh.param<std::string>("/shadow_cast/map_name", "standard");
    int point_num = nh.param<int>("/shadow_cast/point_num", 27000);
    double trans_vel = nh.param<double>("/shadow_cast/trans_vel", 3);
    double rot_vel = nh.param<double>("/shadow_cast/rot_vel", 1.0);
    double angle_min = nh.param<double>("/shadow_cast/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/shadow_cast/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/shadow_cast/angle_incre", M_PI / 360);
    obs_set = nh.param<bool>("/shadow_cast/resquire_no_init_pos", true);
    std::string dev_name = nh.param<std::string>("/shadow_cast/dev_name", "/dev/input/by-id/usb-Keychron_Keychron_K2-event-kbd");

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
    std::vector<Vec2> flattened_points;
    std::vector<char> next_ids;
    printf("Converting from meshes...\n");
    from_cv_points(obstacles, flattened_points, next_ids);
    printf("Mesh flattened.\n");

    printf("Prepare to update point info..., (%lu), (%lu)\n", flattened_points.size(), next_ids.size());
    updatePointInfo(flattened_points.data(), next_ids.data(), flattened_points.size(), false);
    printf("Point info updated.\n");

    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    obs = Eigen::Vector2d(412, 616);
    while (obs_set == false) {
        cv::imshow("disp", src);
        char key = cv::waitKey(10);
        if (key == 27)
            return 0;
    }
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);

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
        Vec3 pose(obs.x(), obs.y(), angle);
        std::vector<Vec2> outputs;
        shadowCasting(pose, outputs);
        draw_from_vec(obstacles, obs, std::move(outputs), src);
        if (render_flag == true) {
            start_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            Eigen::Vector3d act_obs;
            act_obs << obs, angle;       


            end_t = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
            x_motion = 0;
            y_motion = 0;
            time_sum += end_t - start_t;
            time_cnt += 1.0;
            render_flag = false;
        }
        switch(key) {
            case 'w': {
                translation(0) = trans_vel;
                translation(1) = 0;
                render_flag = true;
                delta_angle = 0.0;
                break;
            }
            case 'a': {
                translation(0) = 0;
                translation(1) = trans_vel;
                render_flag = true;
                delta_angle = 0.0;
                break;
            }
            case 's': {
                translation(0) = -trans_vel;
                translation(1) = 0;
                render_flag = true;
                delta_angle = 0.0;
                break;
            }
            case 'd': {
                translation(0) = 0;
                translation(1) = -trans_vel;
                render_flag = true;
                delta_angle = 0.0;
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
        Eigen::Vector2d tmp = obs;
        tmp(0) += cos(angle) * translation(0) + sin(angle) * translation(1);
        tmp(1) += sin(angle) * translation(0) - cos(angle) * translation(1);
        if (occupancy.at<uchar>(int(tmp.y()), int(tmp.x())) > 0) {
            if (occupancy.at<uchar>(int(tmp.y()), int(obs.x())))
                translation(1) = 0.0;
            if (occupancy.at<uchar>(int(obs.y()), int(tmp.x())))
                translation(0) = 0.0;
        }
        x_motion = translation(0);
        y_motion = translation(1);
        obs(0) += cos(angle) * translation(0) + sin(angle) * translation(1);
        obs(1) += sin(angle) * translation(0) - cos(angle) * translation(1);
        translation.setZero();
    }
    double mean_time = time_sum / time_cnt;
    printf("Average running time: %.6lf ms, fps: %.6lf hz\n", mean_time * 1e3, 1.0 / mean_time);
    cv::destroyAllWindows();
    deallocatePoints();
    return 0;
}

