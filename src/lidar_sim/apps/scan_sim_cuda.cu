/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rosbag/bag.h>
#include <ros/ros.h>
#include <chrono>
#include <csignal>
#include <thread>
#include <bitset>
#include "utils/scanUtils.hpp"
#include "utils/keyCtrl.hpp"
#include "utils/mapEdit.h"
#include "utils/consts.h"
#include "utils/vizUtils.hpp"
#include "cuda/host_func.hpp"
#include <iostream>
// #define DEBUG_OUTPUT

cv::Mat src;
Eigen::Vector2d obs, orient, translation, init_obs;
double angle = 0.0;
double delta_angle = 0.0, trans_speed = 2.0, act_speed = 0.0;
bool obs_set = false, mouse_ctrl = true, record_bag = true, initial_saved = false;

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
        if (event == cv::EVENT_LBUTTONDOWN)
            printf("Now angle: %.4lf\n", angle * 180.0 / M_PI);
    } 
    if (event == cv::EVENT_MOUSEWHEEL) {
        int flag = cv::getMouseWheelDelta(flags);
        if (flag < 0)
            trans_speed = std::min(trans_speed + 0.05, 2.0);
        else
            trans_speed = std::max(trans_speed - 0.05, 1.0);
    }
}

std::array<bool, 8> states;
std::array<uchar, 5> trigger;

void controlFlow(char stat) {
    std::array<bool, 5> tmp;
    for (int i = 0; i < 8; i++) {
        if (i < 5)
            tmp[i] = states[i];
        if (stat & (0x01 << i))
            states[i] = true;
        else    
            states[i] = false;
        if (i < 5) {
            if (tmp[i] == true && states[i] == false)   // falling edge
                trigger[i] = 0x01;
            else if (i < 4 && tmp[i] == false && states[i] == true)     // rising edge
                trigger[i] = 0x02;
            else trigger[i] = 0x00;
        }
    }
}

void signalHandler(int sig) {
    cv::destroyAllWindows();
    deallocateFixed();
    deallocateDevice();
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cuda_scan");
    ros::NodeHandle nh;
    cv::setNumThreads(6);
    std::vector<std::vector<cv::Point>> obstacles;
    std::vector<std::vector<Eigen::Vector2d>> meshes;

    std::string name = nh.param<std::string>("/cuda_scan/map_name", "standard");
    std::string scan_topic = nh.param<std::string>("/cuda_scan/scan_topic", "scan");
    std::string bag_name = nh.param<std::string>("/cuda_scan/bag_name", "standard0");
    std::string dev_name = nh.param<std::string>("/cuda_scan/dev_name", "/dev/input/by-id/usb-Keychron_Keychron_K2-event-kbd");
    std::string output_directory = nh.param<std::string>("/cuda_scan/output_directory", "");
    trans_speed = nh.param<double>("/cuda_scan/trans_speed", 4.0);
    double rot_vel = nh.param<double>("/cuda_scan/rot_vel", 1.0);
    double init_x = nh.param<double>("/cuda_scan/init_x", 367.0);
    double init_y = nh.param<double>("/cuda_scan/init_y", 769.0);
    double angle_min = nh.param<double>("/cuda_scan/angle_min", -180.);
    double angle_max = nh.param<double>("/cuda_scan/angle_max", 180.);
    double angle_incre = nh.param<double>("/cuda_scan/angle_incre", 0.25);
    double display_rate = nh.param<double>("/cuda_scan/display_rate", 20.0);

    double lidar_noise = nh.param<double>("/cuda_scan/lidar_noise", 0.02);
    const double pix_resolution = nh.param<double>("/cuda_scan/pix_resolution", 0.02);
    int lidar_multiple = nh.param<int>("/cuda_scan/lidar_multiple", 2);
    bool skip_selection = nh.param<bool>("/cuda_scan/skip_selection", false);
    mouse_ctrl = nh.param<bool>("/cuda_scan/enable_mouse_ctrl", false);
    K_P = nh.param<double>("/cuda_scan/kp", 0.2);
    K_I = nh.param<double>("/cuda_scan/ki", 0.0001);
    K_D = nh.param<double>("/cuda_scan/kd", 0.01);

    ros::Publisher scan_pub, odom_pub, imu_pub;
    scan_pub = nh.advertise<sensor_msgs::LaserScan>("scan", 100);

    const double imu_interval = 1. / display_rate;         // ms
    const double scan_interval = imu_interval * double(lidar_multiple);         // ms

    std::string pack_path = getPackagePath();
    printf("Package prefix: %s\n", pack_path.c_str());
    mapLoad(pack_path + "/../../maps/" + name + ".txt", obstacles);

    src.create(cv::Size(1200, 900), CV_8UC3);
    cv::rectangle(src, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, floors, cv::Scalar(40, 40, 40), -1);
    cv::drawContours(src, obstacles, -1, cv::Scalar(10, 10, 10), -1);
    cv::Mat collision_box;
    cv::cvtColor(src, collision_box, cv::COLOR_BGR2GRAY);
    cv::threshold(collision_box, collision_box, 25, 255, cv::THRESH_BINARY_INV);
    cv::dilate(collision_box, collision_box, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    states.fill(false);
    trigger.fill(false);
    for (const Obstacle& egs: obstacles) {
        cv::circle(src, egs.front(), 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(src, egs.back(), 3, cv::Scalar(255, 0, 0), -1);
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    signal(SIGINT, signalHandler);
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
    int scan_cnt = 0;
    // TODO: 此处需要修正激光雷达角度
    Eigen::Vector3d angles(angle_min * M_PI / 180., angle_max * M_PI / 180., angle_incre * M_PI / 180. / 3.);
    angles.x() += angles.z() / 2.;
    angles.y() -= angles.z() / 2.;
    Eigen::Vector3d actual_lidar(angles.x() + angles.z(), angles.y() - angles.z(), angles.z() * 3.);
    int ray_num = static_cast<int>(round((angles.y() - angles.x()) / angles.z())) + 1, actual_ray_num = ray_num / 3;
    if (ray_num % 360) {
        std::cerr << "Ray num must be the multiple of 360. Now it is: " << ray_num << std::endl;
        return -1; 
    }

    printf("Intializing CUDA device...\n");
    intializeFixed(ray_num);
    cvPoints2Meshes(obstacles, meshes);
    unwrapMeshes(meshes);
    printf("CUDA device intialized.\n");

    std::vector<Eigen::Vector3d> gtt;       // ground truth tragectory
    std::string sub_fix;
    if (lidar_noise < 0.05)
        sub_fix = "";
    else
        sub_fix = "h";
    std::string output_file_name = bag_name + "_" + toString(angle_max - angle_min) + "_" + toString(angle_incre) + sub_fix;
    std::string screen_shot_outpath;
    std::string output_bag_path;
    if (output_directory.length()) {
        output_bag_path = output_directory + output_file_name + "/data.bag";
        screen_shot_outpath = output_directory + output_file_name + "/loc_initial.png";
    } else {
        screen_shot_outpath = pack_path + "/../../bags/" + "loc_initial.png";
        output_bag_path = pack_path + "/../../bags/" + output_file_name + ".bag";
    }
    rosbag::Bag bag(output_bag_path, rosbag::bagmode::Write);
    TicToc timer, render_timer;
    KeyCtrl kc(dev_name);
    std::thread worker(&KeyCtrl::onKeyThread, &kc);
    worker.detach();
    double render_sum = 0.0, render_cnt = 0.0;
    ros::Rate rate(display_rate);
    timer.tic();
    while (ros::ok()) {
        drawScannerPose(obs, angle, src);
        cv::imshow("disp", src);
        if (initial_saved == false) {
            initial_saved = true;
            cv::imwrite(screen_shot_outpath, src);
        }
        cv::waitKey(1);
        char key = status;
        controlFlow(key);
        record_bag = bool(trigger.back()) ^ record_bag;
        bool collided = false;
        std::vector<double> range;
        std::vector<float> ranges(actual_ray_num);
        render_sum += rayTraceRenderCpp(angles, obs, angle, ray_num, ranges);
        render_cnt += 1.0;
        drawObstacles(obstacles, obs, src);
        drawScanLine(obs, actual_lidar, ranges, angle, src);
        
        plotSpeedInfo(src, trans_speed, act_speed);
        Eigen::Vector3d tmp_pose;
        tmp_pose << obs.x(), obs.y(), angle;
        Eigen::Vector2d tmp = obs + translation;

        if (collision_box.at<uchar>(int(tmp.y()), int(tmp.x())) > 0) {
            if (collision_box.at<uchar>(int(tmp.y()), int(obs.x())))
                translation(1) = 0.0;
            if (collision_box.at<uchar>(int(obs.y()), int(tmp.x())))
                translation(0) = 0.0;
            collided = true;
        }
        if (record_bag == true)
            cv::circle(src, cv::Point(15, 15), 10, cv::Scalar(0, 0, 255), -1);
        obs += translation;
        tf::StampedTransform gt_tf;
        makeTransform(Eigen::Vector3d(obs.x() * pix_resolution, obs.y() * pix_resolution, angle), "map", scan_topic, gt_tf);
        bool running = states[0] | states[1] | states[2] | states[3];
        if (running && !collided) {
            act_speed = (0.5 * trans_speed + 0.5 * act_speed);
            if (std::abs(act_speed - trans_speed) < 1e-5)
                act_speed = trans_speed;
        } else {
            act_speed *= 0.5;
            if (act_speed < 1e-5)
                act_speed = 0.0;
        }
        
        timer.tic();
        // odom_tf.
        sendStampedTranform(gt_tf);

        ros::Time now_stamp = ros::Time::now();
        if (record_bag == true) {
            // sendStampedTranform(scan_tf);
            scan_cnt += 1;
            if (scan_cnt >= lidar_multiple) {
                scan_cnt = 0;
                // sensor_msgs::LaserScan scan;
                tf::tfMessage gt_tfmsg, scan_tfmsg;
                // makeScan(range, angles, scan, scan_topic, scan_interval);
                stampedTransform2TFMsg(gt_tf, gt_tfmsg);
                // odometry data used for LiDAR should not be biased by init_obs
                // Consider that init_angle is always (if the code is not modified) 0., angle is unbiased
                bag.write("tf", now_stamp, gt_tfmsg);
                // bag.write(scan_topic, now_stamp, scan);
                // scan_pub.publish(scan);
            }
        }
        translation.setZero();
        delta_angle = 0.0;
        if (states[0] == true) {
            translation(0) += cos(angle) * act_speed;
            translation(1) += sin(angle) * act_speed;
        }
        if (states[1] == true) {
            translation(0) += sin(angle) * act_speed;
            translation(1) += -cos(angle) * act_speed;
        }
        if (states[2] == true) {
            translation(0) += -cos(angle) * act_speed;
            translation(1) += -sin(angle) * act_speed;
        }
        if (states[3] == true) {
            translation(0) += -sin(angle) * act_speed;
            translation(1) += cos(angle) * act_speed;
        }
        if (mouse_ctrl == false && states[5] == true) {
            angle -= rot_vel;
            delta_angle = -rot_vel;
            if (angle < -M_PI)
                angle += 2 * M_PI;
        }
        if (mouse_ctrl == false && states[6] == true) {
            angle += rot_vel;
            delta_angle = rot_vel;
            if (angle > M_PI)
                angle -= 2 * M_PI;
        }
        if (states[7] == true) {
            break;
        }
        if (mouse_ctrl == true) {
            delta_angle = pidAngle(orient, obs, angle);
            angle += delta_angle;
            if (angle > M_PI)
                angle -= 2 * M_PI;
            else if (angle < -M_PI)
                angle += 2 * M_PI;
        }
        rate.sleep();
    }
    #ifdef DEBUG_OUTPUT
        printf("Kinetic information is saved to 'bags/kinetic.log', odom information is saved to 'bags/odom.log'\n");
        kinetic_info.close();
        odom_info.close();
    #endif 
    bag.close();
    cv::destroyAllWindows();
    deallocateFixed();
    deallocateDevice();
    printf("average render time: %lf ms\n", render_sum / render_cnt);
    return 0;
}
