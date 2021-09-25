#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rosbag/bag.h>
#include <ros/ros.h>
#include <chrono>
#include "scanUtils.hpp"
#include "lidarSim.hpp"
#include "mapEdit.h"
#include "consts.h"

cv::Mat src;
Eigen::Vector2d obs, orient, translation, init_obs, total_trans;
double angle = 0.0, angle_sum = 0.0;
double delta_angle = 0.0;
bool obs_set = false, mouse_ctrl = false, record_bag = false;

void on_mouse(int event, int x, int y, int flags, void *ustc) {
    if (event == cv::EVENT_LBUTTONDOWN && obs_set == false) {
        printf("cv::Point(%d, %d),\n", x, y);
        obs(0) = double(x);
        obs(1) = double(y);
        cv::circle(src, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        obs_set = true;
    } else if (obs_set == true) {
        orient(0) = double(x);
        orient(1) = double(y);
        if (event == cv::EVENT_LBUTTONDOWN) {
            printf("Now angle: %.4lf\n", angle * 180.0 / M_PI);
        }
    } 
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan");
    ros::NodeHandle nh;
    cv::setNumThreads(4);
    std::vector<std::vector<cv::Point>> obstacles;
    std::string name = nh.param<std::string>("/scan/map_name", "standard");
    double trans_speed = nh.param<double>("/scan/trans_speed", 4.0);
    double rot_vel = nh.param<double>("/scan/rot_vel", 1.0);
    double init_x = nh.param<double>("/scan/init_x", 367.0);
    double init_y = nh.param<double>("/scan/init_y", 769.0);
    double angle_min = nh.param<double>("/scan/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/scan/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/scan/angle_incre", M_PI / 1800.0);
    double fps = nh.param<double>("/scan/lidar_fps", 20.0);
    double translation_noise = nh.param<double>("/scan/translation_noise", 0.08);
    double rotation_noise = nh.param<double>("/scan/rotation_noise", 0.01);
    double lidar_noise = nh.param<double>("/scan/lidar_noise", 0.02);
    bool skip_selection = nh.param<bool>("/scan/skip_selection", false);
    bool direct_pub = nh.param<bool>("/scan/direct_pub", false);
    ros::Publisher scan_pub, odom_pub;
    if (direct_pub == true) {
        scan_pub = nh.advertise<sensor_msgs::LaserScan>("scan", 100);
        odom_pub = nh.advertise<nav_msgs::Odometry>("sim_odom", 100);
    }
    K_P = nh.param<double>("/scan/kp", 0.2);
    K_I = nh.param<double>("/scan/ki", 0.0001);
    K_D = nh.param<double>("/scan/kd", 0.01);
    const Eigen::Vector2d noise(translation_noise, rotation_noise);

    double msg_interval = 1000.0 / fps;         // ms

    std::string pack_path = getPackagePath();
    printf("Package prefix: %s\n", pack_path.c_str());
    mapLoad(pack_path + "/../../maps/" + name + ".txt", obstacles);
    src.create(cv::Size(1200, 900), CV_8UC3);
    cv::rectangle(src, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, floors, cv::Scalar(40, 40, 40), -1);
    cv::drawContours(src, obstacles, -1, cv::Scalar(10, 10, 10), -1);
    
    for (const Obstacle& egs: obstacles) {
        cv::circle(src, egs.front(), 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(src, egs.back(), 3, cv::Scalar(255, 0, 0), -1);
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    obs = Eigen::Vector2d(init_x, init_y);
    init_obs = Eigen::Vector2d(init_x, init_y);
    rot_vel = rot_vel * M_PI / 180.0;
    if (skip_selection == false) {
        while (obs_set == false) {
            cv::imshow("disp", src);
            char key = cv::waitKey(10);
            if (key == 27)
                return 0;
        }
    }
    init_obs = obs;
    // __pose__ = 
    bool render_flag = true;
    double time_cnt = 1.0, time_sum = 0.0, bag_time_sum = 0.0;
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    LidarSim ls(angles, lidar_noise);
    std::vector<Eigen::Vector3d> gtt;       // ground truth tragectory
    rosbag::Bag bag(pack_path + "/../../bags/" + name + ".bag", rosbag::bagmode::Write);
    nav_msgs::Odometry odom;
    TicToc timer;
    while (true) {
        cv::imshow("disp", src);
        char key = cv::waitKey(1);
        bool break_flag = false;
        if (render_flag == true) {
            timer.tic();
            std::vector<double> range;
            ls.scan(obstacles, obs, range, src, angle);
            time_sum += timer.toc();
            tf::StampedTransform stamped_tf;
            makeTransform(Eigen::Vector3d(obs.x() * 0.02, obs.y() * 0.02, angle), "map", "scan", stamped_tf);
            if (record_bag == true) {
                total_trans += translation;
                angle_sum += delta_angle;
                sendStampedTranform(stamped_tf);
                bag_time_sum += timer.toc();
                cv::circle(src, cv::Point(15, 15), 10, cv::Scalar(0, 0, 255), -1);
            }
            time_cnt += 1.0;
            render_flag = false;
            obs += translation;
            if (bag_time_sum > msg_interval && record_bag == true) {
                bag_time_sum = 0.0;
                nav_msgs::Odometry odom;
                sensor_msgs::LaserScan scan;
                tf::tfMessage tf_msg;
                Eigen::Vector3d delta_pose;
                delta_pose << total_trans, angle_sum;
                makeScan(range, angles, scan, "scan", msg_interval / 1e3);
                stampedTransform2TFMsg(stamped_tf, tf_msg);
                makePerturbedOdom(delta_pose, odom, noise, "map", "scan");
                bag.write("scan", ros::Time::now(), scan);
                bag.write("sim_tf", ros::Time::now(), tf_msg);
                bag.write("sim_odom", ros::Time::now(), odom);
                if (direct_pub == true) {
                    odom_pub.publish(odom);
                    scan_pub.publish(scan);
                }
                total_trans.setZero();
                angle_sum = 0.0;
            }
            translation.setZero();
        }
        switch(key) {
            case 'w': {
                translation(0) = cos(angle) * trans_speed;
                translation(1) = sin(angle) * trans_speed;
                render_flag = true;
                break;
            }
            case 'a': {
                translation(0) = sin(angle) * trans_speed;
                translation(1) = -cos(angle) * trans_speed;
                render_flag = true;
                break;
            }
            case 's': {
                translation(0) = -cos(angle) * trans_speed;
                translation(1) = -sin(angle) * trans_speed;
                render_flag = true;
                break;
            }
            case 'd': {
                translation(0) = -sin(angle) * trans_speed;
                translation(1) = cos(angle) * trans_speed;
                render_flag = true;
                break;
            }
            case 'p': {
                if (mouse_ctrl == false)
                    angle += rot_vel;
                if (angle > M_PI)
                    angle -= 2 * M_PI;
                render_flag = true;
                break;
            }
            case 'o': {
                if (mouse_ctrl == false)
                    angle -= rot_vel;
                if (angle < -M_PI)
                    angle += 2 * M_PI;
                render_flag = true;
                break;
            }
            case 'm': {
                if (mouse_ctrl == false)
                    printf("Mouse angle control is on.\n");
                else 
                    printf("Mouse angle control is off.\n");
                mouse_ctrl = !mouse_ctrl;
                break;
            }
            case 'r': {
                record_bag = !record_bag;
                if (record_bag == true) {
                    __pose__ << obs * 0.02, angle;
                }
                bag_time_sum = 0.0;
                break;
            }
            case 27: break_flag = true;
        }
        if (mouse_ctrl == true) {
            delta_angle = pidAngle(orient, obs, angle);
            angle += delta_angle;
            if (angle > M_PI)
                angle -= 2 * M_PI;
            else if (angle < -M_PI)
                angle += 2 * M_PI;
            render_flag = true;
        }
        if (break_flag == true)
            break;
    }
    bag.close();
    double mean_time = time_sum / time_cnt;
    printf("Average running time: %.6lf ms, fps: %.6lf hz\n", mean_time, 1000.0 / mean_time);
    cv::destroyAllWindows();
    return 0;
}