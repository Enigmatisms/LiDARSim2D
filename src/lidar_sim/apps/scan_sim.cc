/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rosbag/bag.h>
#include <ros/ros.h>
#include <chrono>
#include <thread>
#include <bitset>
#include "utils/scanUtils.hpp"
#include "utils/keyCtrl.hpp"
#include "utils/mapEdit.h"
#include "utils/consts.h"
#include "volume/lidarSim.hpp"
#include <iostream>
#define DEBUG_OUTPUT

cv::Mat src;
Eigen::Vector2d obs, orient, translation, init_obs;
double angle = 0.0;
double delta_angle = 0.0, trans_speed = 2.0, act_speed = 0.0;
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

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan");
    ros::NodeHandle nh;
    cv::setNumThreads(4);
    std::vector<std::vector<cv::Point>> obstacles;


    std::string name = nh.param<std::string>("/scan/map_name", "standard");
    std::string scan_topic = nh.param<std::string>("/scan/scan_topic", "scan");
    std::string odom_topic = nh.param<std::string>("/scan/odom_topic", "odom");
    std::string imu_topic = nh.param<std::string>("/scan/odom_topic", "imu");
    std::string bag_name = nh.param<std::string>("/scan/bag_name", "standard");
    std::string dev_name = nh.param<std::string>("/scan/dev_name", "/dev/input/by-id/usb-Keychron_Keychron_K2-event-kbd");
    trans_speed = nh.param<double>("/scan/trans_speed", 4.0);
    double rot_vel = nh.param<double>("/scan/rot_vel", 1.0);
    double init_x = nh.param<double>("/scan/init_x", 367.0);
    double init_y = nh.param<double>("/scan/init_y", 769.0);
    double angle_min = nh.param<double>("/scan/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/scan/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/scan/angle_incre", M_PI / 1800.0);
    double display_rate = nh.param<double>("/scan/display_rate", 20.0);

    double translation_noise = nh.param<double>("/scan/translation_noise", 0.08);
    double rotation_noise = nh.param<double>("/scan/rotation_noise", 0.01);
    double trans_vel_noise = nh.param<double>("/scan/trans_vel_noise", 0.01);
    double rot_vel_noise = nh.param<double>("/scan/rot_vel_noise", 0.01);

    double lidar_noise = nh.param<double>("/scan/lidar_noise", 0.02);
    const double pix_resolution = nh.param<double>("/scan/pix_resolution", 0.02);
    int lidar_multiple = nh.param<int>("/scan/lidar_multiple", 2);
    bool skip_selection = nh.param<bool>("/scan/skip_selection", false);
    bool imu_plot = nh.param<bool>("/scan/imu_plot", false);
    bool use_recorded_path = nh.param<bool>("/scan/use_recorded_path", false);
    bool bag_imu = nh.param<bool>("/scan/bag_imu", false);
    bool bag_odom = nh.param<bool>("/scan/bag_odom", false);
    mouse_ctrl = nh.param<bool>("/scan/enable_mouse_ctrl", false);
    K_P = nh.param<double>("/scan/kp", 0.2);
    K_I = nh.param<double>("/scan/ki", 0.0001);
    K_D = nh.param<double>("/scan/kd", 0.01);


    ros::Publisher scan_pub, odom_pub, imu_pub;
    scan_pub = nh.advertise<sensor_msgs::LaserScan>("scan", 100);
    odom_pub = nh.advertise<nav_msgs::Odometry>("sim_odom", 100);
    imu_pub = nh.advertise<sensor_msgs::Imu>("sim_imu", 100);
    const Eigen::Vector4d noise(translation_noise, rotation_noise, trans_vel_noise, rot_vel_noise);

    const double scan_interval = 1. / display_rate * double(lidar_multiple);         // ms

    std::string pack_path = getPackagePath();
    printf("Package prefix: %s\n", pack_path.c_str());
    mapLoad(pack_path + "/../../maps/" + name + ".txt", obstacles);
    std::ofstream* file = nullptr;
    if (imu_plot)
        file = new std::ofstream(pack_path + "/../../data/data.txt", std::ios::out);
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
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    LidarSim ls(angles, lidar_noise);
    std::vector<Eigen::Vector3d> gtt;       // ground truth tragectory
    rosbag::Bag bag(pack_path + "/../../bags/" + bag_name + ".bag", rosbag::bagmode::Write);
    nav_msgs::Odometry odom;
    TicToc timer, inner_timer;
    std::atomic_char status = 0x00;
    KeyCtrl kc(dev_name, status);
    std::thread worker(&KeyCtrl::onKeyThread, &kc);
    tf::StampedTransform odom_tf;
    double init_angle = angle;
    worker.detach();

    std::ifstream recorded_path;
    std::ofstream record_output;
    std::vector<Eigen::Vector4d> path_vec;
    if (use_recorded_path == true) {
        recorded_path.open(pack_path + "/../../bags/path.txt", std::ios::in);
        std::string line;
        while (getline(recorded_path, line)) {
            std::stringstream ss;
            ss << line;
            Eigen::Vector4d pos;
            ss >> pos(0) >> pos(1) >> pos(2) >> pos(3);
            path_vec.push_back(pos);
        }
        recorded_path.close();
    } else {
        record_output.open(pack_path + "/../../bags/path.txt", std::ios::out);
    }
    if (use_recorded_path)
        record_bag = true;
    size_t path_counter = 0;
    #ifdef DEBUG_OUTPUT
        std::fstream kinetic_info;
        kinetic_info.open(pack_path + "/../../bags/kinetic.log", std::ios::out);
    #endif
    inner_timer.tic();
    ros::Rate rate(display_rate);
    while (ros::ok()) {
        cv::Point start(obs.x(), obs.y()), end(obs.x() + 20 * cos(angle), obs.y() + 20 * sin(angle));
        cv::arrowedLine(src, start, end, cv::Scalar(255, 0, 0), 2);
        cv::imshow("disp", src);
        cv::waitKey(1);
        char key = status;
        controlFlow(key);
        record_bag = bool(trigger.back()) ^ record_bag;
        bool collided = false;
        std::vector<double> range;
        if (use_recorded_path == true) {
            if (path_counter >= path_vec.size()) {
                printf("No more new pose, exiting...\n");
                break;
            }
            obs = path_vec[path_counter].block<2, 1>(1, 0);
            angle = path_vec[path_counter](3);
        }
        ls.scan(obstacles, obs, range, src, angle);
        plotSpeedInfo(src, trans_speed, act_speed);
        Eigen::Vector3d tmp_pose;
        if (use_recorded_path == false)
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
        double act_trans_x = 0.0, act_trans_y = 0.0;            // act_trans(相对) 与 translation(绝对) 是完全不同的概念
        if (states[0] == true)
            act_trans_x = act_speed;
        else if (states[2] == true)
            act_trans_x = -act_speed;
        if (states[1] == true)
            act_trans_y = -act_speed;
        else if (states[3] == true)
            act_trans_y = act_speed;
        if (collided)
            act_trans_x = act_trans_y = 0.0;
        Eigen::Vector2d imu_trans(act_trans_x * pix_resolution, act_trans_y * pix_resolution);
        Eigen::Vector3d delta_pose;
        delta_pose << imu_trans, delta_angle;
        
        odomTFSimulation(noise, delta_pose, odom_tf, odom_topic, scan_topic);
        sensor_msgs::Imu imu_msg;
        nav_msgs::Odometry odom;
        double duration = makeImuMsg(imu_trans, scan_topic, angle, imu_msg, file);
        makePerturbedOdom(noise, init_obs * pix_resolution, delta_pose, odom, init_angle, duration, odom_topic, scan_topic);
        // odom_tf.
        tf::Vector3 odom_translation = odom_tf.getOrigin();
        printf("%lf, %lf, %lf, %lf, %lf\n", delta_pose.x(), delta_pose.y(), delta_pose.z(), odom_translation.x(), odom_translation.y());
        sendStampedTranform(gt_tf);
        tf::StampedTransform odom2map = getOdom2MapTF(gt_tf, odom_tf, init_obs * pix_resolution, init_angle);
        sendStampedTranform(odom2map);

        odom_pub.publish(odom);
        imu_pub.publish(imu_msg);       // IMU will be published unconditionally
        if (record_bag == true) {
            // sendStampedTranform(scan_tf);
            scan_cnt += 1;
            if (scan_cnt >= lidar_multiple) {
                scan_cnt = 0;
                sensor_msgs::LaserScan scan;
                tf::tfMessage gt_tfmsg, scan_tfmsg;
                makeScan(range, angles, scan, scan_topic, scan_interval);
                stampedTransform2TFMsg(gt_tf, gt_tfmsg);
                #ifdef DEBUG_OUTPUT
                    if (record_bag == true) {
                        tf::Vector3 p = gt_tf.getOrigin();
                        kinetic_info << translation.x() << " " << translation.y() << " " << delta_angle << " " << p.x() * 50. << " " << obs.x() << " " << p.y() * 50. << " " << obs.y() << " " <<  angle << " " << act_speed << " " << (running & !collided) << std::endl;
                    }
                #endif
                bag.write("tf", ros::Time::now(), gt_tfmsg);
                bag.write(scan_topic, ros::Time::now(), scan);
                scan_pub.publish(scan);

                ros::Time now_stamp = ros::Time::now();
                if (use_recorded_path)
                    ros::Time now_stamp = now_stamp.fromNSec(path_vec[path_counter](0));
                if (use_recorded_path == false) {
                    record_output << now_stamp.toNSec() << " " << tmp_pose(0) << " " << tmp_pose(1) << " " << tmp_pose(2) << std::endl;
                }
            }
            
            if (bag_imu)
                bag.write(imu_topic, ros::Time::now(), imu_msg);
            if (bag_odom)
                bag.write(odom_topic, ros::Time::now(), odom);
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
        printf("Kinetic information is saved to 'bags/kinetic.log'\n");
        kinetic_info.close();
    #endif 
    bag.close();
    if (imu_plot == true) {
        file->close();
        delete file;
    }
    if (use_recorded_path == false) {
        record_output.close();
        printf("Trajectory record completed.\n");
    }
    cv::destroyAllWindows();
    return 0;
}
