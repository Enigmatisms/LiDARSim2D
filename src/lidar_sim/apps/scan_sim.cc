#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rosbag/bag.h>
#include <ros/ros.h>
#include <chrono>
#include <thread>
#include <bitset>
#include "scanUtils.hpp"
#include "lidarSim.hpp"
#include "keyCtrl.hpp"
#include "mapEdit.h"
#include "consts.h"

cv::Mat src;
Eigen::Vector2d obs, orient, translation, init_obs, total_trans;
double angle = 0.0, angle_sum = 0.0;
double delta_angle = 0.0, trans_speed = 2.0;
bool obs_set = false, mouse_ctrl = true, record_bag = false;

void on_mouse(int event, int x, int y, int flags, void *ustc) {
    if (event == cv::EVENT_LBUTTONDOWN && obs_set == false) {
        printf("cv::Point(%d, %d),\n", x, y);
        obs(0) = double(x);
        obs(1) = double(y);
        cv::circle(src, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        obs_set = true;
    } else if (mouse_ctrl == true && obs_set == true) {
        orient(0) = double(x);
        orient(1) = double(y);
        if (event == cv::EVENT_LBUTTONDOWN) {
            printf("Now angle: %.4lf\n", angle * 180.0 / M_PI);
        }
    } 
    if (event == cv::EVENT_MOUSEWHEEL) {
        int flag = cv::getMouseWheelDelta(flags);
        if (flag < 0) {
            trans_speed = std::min(trans_speed + 0.05, 2.0);
        } else {
            trans_speed = std::max(trans_speed - 0.05, 1.0);
        }
    }
}

std::array<bool, 8> states;

bool controlFlow(char stat) {
    bool old_val = states[6];
    for (int i = 0; i < 8; i++) {
        if (stat & (0x01 << i))
            states[i] = true;
        else    
            states[i] = false;
    }
    return ((old_val == true) && (states[6] == false));
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan");
    ros::NodeHandle nh;
    cv::setNumThreads(4);
    std::vector<std::vector<cv::Point>> obstacles;
    std::string name = nh.param<std::string>("/scan/map_name", "standard");
    std::string dev_name = nh.param<std::string>("/scan/dev_name", "/dev/input/by-id/usb-Keychron_Keychron_K2-event-kbd");
    trans_speed = nh.param<double>("/scan/trans_speed", 4.0);
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
    mouse_ctrl = nh.param<bool>("/scan/enable_mouse_ctrl", false);
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
    /// collision box
    cv::Mat collision_box;
    cv::cvtColor(src, collision_box, cv::COLOR_BGR2GRAY);
    cv::threshold(collision_box, collision_box, 25, 255, cv::THRESH_BINARY_INV);
    cv::dilate(collision_box, collision_box, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    for (int i = 0; i < 8; i++)
        states[i] = false;
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
    double time_cnt = 1.0, time_sum = 0.0, bag_time_sum = 0.0;
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    LidarSim ls(angles, lidar_noise);
    std::vector<Eigen::Vector3d> gtt;       // ground truth tragectory
    rosbag::Bag bag(pack_path + "/../../bags/" + name + ".bag", rosbag::bagmode::Write);
    nav_msgs::Odometry odom;
    TicToc timer;
    std::atomic_char status = 0x00;
    KeyCtrl kc(dev_name, status);
    std::thread worker(&KeyCtrl::onKeyThread, &kc);
    worker.detach();
    while (true) {
        cv::imshow("disp", src);
        cv::waitKey(5);
        char key = status;
        bool edge_down = controlFlow(key);
        record_bag = edge_down ^ record_bag;
        bool break_flag = false;
        timer.tic();
        std::vector<double> range;
        ls.scan(obstacles, obs, range, src, angle);
        plotSpeedInfo(src, trans_speed);
        time_sum += timer.toc();
        tf::StampedTransform stamped_tf;
        makeTransform(Eigen::Vector3d(obs.x() * 0.02, obs.y() * 0.02, angle), "map", "scan", stamped_tf);
        Eigen::Vector2d tmp = obs + translation;
        if (collision_box.at<uchar>(int(tmp.y()), int(tmp.x())) > 0) {
            if (collision_box.at<uchar>(int(tmp.y()), int(obs.x())))
                translation(1) = 0.0;
            if (collision_box.at<uchar>(int(obs.y()), int(tmp.x())))
                translation(0) = 0.0;
        }
        if (record_bag == true) {
            total_trans += translation;
            angle_sum += delta_angle;
            sendStampedTranform(stamped_tf);
            bag_time_sum += timer.toc();
            cv::circle(src, cv::Point(15, 15), 10, cv::Scalar(0, 0, 255), -1);
        }
        time_cnt += 1.0;
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
        if (states[0] == true) {
            translation(0) += cos(angle) * trans_speed;
            translation(1) += sin(angle) * trans_speed;
        }
        if (states[1] == true) {
            translation(0) += sin(angle) * trans_speed;
            translation(1) += -cos(angle) * trans_speed;
        }
        if (states[2] == true) {
            translation(0) += -cos(angle) * trans_speed;
            translation(1) += -sin(angle) * trans_speed;
        }
        if (states[3] == true) {
            translation(0) += -sin(angle) * trans_speed;
            translation(1) += cos(angle) * trans_speed;
        }
        if (mouse_ctrl == false && states[4] == true) {
            angle -= rot_vel;
            if (angle < -M_PI)
                angle += 2 * M_PI;
        }
        if (mouse_ctrl == false && states[5] == true) {
            angle += rot_vel;
            if (angle > M_PI)
                angle -= 2 * M_PI;
        }
        if (states[7] == true) {
            break;
        }
        if (edge_down == true) {
            printf("Bag time reset.\n");
            bag_time_sum = 0.0;
        }
        if (mouse_ctrl == true) {
            delta_angle = pidAngle(orient, obs, angle);
            angle += delta_angle;
            if (angle > M_PI)
                angle -= 2 * M_PI;
            else if (angle < -M_PI)
                angle += 2 * M_PI;
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
