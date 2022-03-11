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
#include "utils/joyCtrl.hpp"
#include "utils/mapEdit.h"
#include "utils/consts.h"
#include "volume/lidarSim.hpp"

cv::Mat src;
Eigen::Vector2d obs, translation, init_obs;
double angle = 0.0;
double delta_angle = 0.0, trans_speed_amp = 3.0, rot_vel_amp = 1.5, act_speed_x = 0.0, act_speed_y = 0.0;
bool obs_set = false, mouse_ctrl = true, record_bag = false, exit_flag = false;

void on_mouse(int event, int x, int y, int flags, void *ustc) {
    if (event == cv::EVENT_LBUTTONDOWN && obs_set == false) {
        printf("cv::Point(%d, %d),\n", x, y);
        obs(0) = double(x);
        obs(1) = double(y);
        cv::circle(src, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        obs_set = true;
    } else if (mouse_ctrl == true && obs_set == true) {
        if (event == cv::EVENT_LBUTTONDOWN)
            printf("Now angle: %.4lf\n", angle * 180.0 / M_PI);
    }
}

std::unique_ptr<JoyCtrl> joyc;
float speed[3] = {0, 0, 0};

void controlFlow() {
    speed[0] = reinterpret_cast<float &>(joyc->forward_speed) * trans_speed_amp;
    speed[1] = reinterpret_cast<float &>(joyc->lateral_speed) * trans_speed_amp;
    speed[2] = -reinterpret_cast<float &>(joyc->angular_vel) * rot_vel_amp;
    record_bag = joyc->record_bag;
    exit_flag = joyc->exit_flag;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan_joy");
    ros::NodeHandle nh;
    cv::setNumThreads(4);
    std::vector<std::vector<cv::Point>> obstacles;
    std::string name = nh.param<std::string>("/scan_joy/map_name", "standard");
    std::string bag_name = nh.param<std::string>("/scan_joy/bag_name", "standard");
    trans_speed_amp = nh.param<double>("/scan_joy/trans_speed_amp", 3.0);
    rot_vel_amp = nh.param<double>("/scan_joy/rot_vel_amp", 1.5);
    double init_x = nh.param<double>("/scan_joy/init_x", 367.0);
    double init_y = nh.param<double>("/scan_joy/init_y", 769.0);
    double angle_min = nh.param<double>("/scan_joy/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/scan_joy/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/scan_joy/angle_incre", M_PI / 1800.0);
    double lidar_fps = nh.param<double>("/scan_joy/lidar_fps", 20.0);
    double odom_fps = nh.param<double>("/scan_joy/odom_fps", 40.0);

    double translation_noise = nh.param<double>("/scan_joy/translation_noise", 0.08);
    double rotation_noise = nh.param<double>("/scan_joy/rotation_noise", 0.01);
    double trans_vel_noise = nh.param<double>("/scan_joy/trans_vel_noise", 0.01);
    double rot_vel_noise = nh.param<double>("/scan_joy/rot_vel_noise", 0.01);

    double lidar_noise = nh.param<double>("/scan_joy/lidar_noise", 0.02);
    bool skip_selection = nh.param<bool>("/scan_joy/skip_selection", false);
    bool imu_plot = nh.param<bool>("/scan_joy/imu_plot", false);
    bool use_recorded_path = nh.param<bool>("/scan_joy/use_recorded_path", false);
    bool show_ray = nh.param<bool>("/scan_joy/show_ray", false);
    mouse_ctrl = nh.param<bool>("/scan_joy/enable_mouse_ctrl", false);
    ros::Publisher scan_pub, odom_pub, imu_pub;
    scan_pub = nh.advertise<sensor_msgs::LaserScan>("scan", 100);
    odom_pub = nh.advertise<nav_msgs::Odometry>("sim_odom", 100);
    imu_pub = nh.advertise<sensor_msgs::Imu>("sim_imu", 100);
    const Eigen::Vector4d noise(translation_noise, rotation_noise, trans_vel_noise, rot_vel_noise);

    const double scan_interval = 1000.0 / lidar_fps;         // ms
    const double odom_interval = 1000.0 / odom_fps;         // ms

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
    for (const Obstacle& egs: obstacles) {
        cv::circle(src, egs.front(), 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(src, egs.back(), 3, cv::Scalar(255, 0, 0), -1);
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    cv::namedWindow("disp", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("disp", on_mouse, NULL);
    obs = Eigen::Vector2d(init_x, init_y);
    init_obs = Eigen::Vector2d(init_x, init_y);
    rot_vel_amp = rot_vel_amp * M_PI / 180.0;
    if (skip_selection == false) {
        while (obs_set == false) {
            cv::imshow("disp", src);
            char key = cv::waitKey(10);
            if (key == 27)
                return 0;
        }
    }
    init_obs = obs;
    double time_cnt = 1.0, time_sum = 0.0, scan_time_sum = 0.0, odom_time_sum = 0.0;
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    LidarSim ls(angles, lidar_noise);
    std::vector<Eigen::Vector3d> gtt;       // ground truth tragectory
    rosbag::Bag bag(pack_path + "/../../bags/" + bag_name + ".bag", rosbag::bagmode::Write);
    nav_msgs::Odometry odom;
    TicToc timer;
    std::atomic_char status = 0x00;
    tf::StampedTransform odom_tf;
    double init_angle = angle;
    joyc = std::unique_ptr<JoyCtrl>(new JoyCtrl);
    joyc->start(nh);

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
    int key_wait = (use_recorded_path == true) ? 15 : 3;
    while (true) {
        cv::Point start(obs.x(), obs.y()), end(obs.x() + 20 * cos(angle), obs.y() + 20 * sin(angle));
        cv::arrowedLine(src, start, end, cv::Scalar(255, 0, 0), 2);
        cv::imshow("disp", src);
        cv::waitKey(key_wait);
        char key = status;
        controlFlow();
        bool collided = false;
        timer.tic();
        std::vector<double> range;
        if (use_recorded_path == true) {
            if (path_counter >= path_vec.size()) {
                printf("No more new pose, exiting...\n");
                break;
            }
            obs = path_vec[path_counter].block<2, 1>(1, 0);
            angle = path_vec[path_counter](3);
        }
        ls.scan(obstacles, obs, range, src, angle, show_ray);
        plotSpeedInfo(src, trans_speed_amp, act_speed_x);
        time_sum += timer.toc();
        tf::StampedTransform gt_tf, scan_tf;
        makeTransform(Eigen::Vector3d(init_obs.x() * 0.02, init_obs.y() * 0.02, init_angle), "map", "odom", odom_tf);
        makeTransform(Eigen::Vector3d(obs.x() * 0.02, obs.y() * 0.02, angle), "map", "scan_gt", gt_tf);
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
        if (record_bag == true) {
            scan_time_sum += timer.toc();
            odom_time_sum += timer.toc();
            cv::circle(src, cv::Point(15, 15), 10, cv::Scalar(0, 0, 255), -1);
        }
        time_cnt += 1.0;
        obs += translation;
        bool running = abs(speed[0]) > 1e-5;
        if (running && !collided) {
            act_speed_x = speed[0];
            act_speed_y = speed[1];
        } else {
            act_speed_x *= 0.5;
            act_speed_y *= 0.5;
            if (abs(act_speed_x) < 1e-5)
                act_speed_x = 0.0;
            if (abs(act_speed_y) < 1e-5)
                act_speed_y = 0.0;
        }
        double act_trans_x = 0.0, act_trans_y = 0.0;            // act_trans(相对) 与 translation(绝对) 是完全不同的概念
        if (collided)
            act_trans_x = act_trans_y = 0.0;
        Eigen::Vector2d imu_trans(act_trans_x * 0.02, act_trans_y * 0.02);
        Eigen::Vector3d delta_pose;
        delta_pose << imu_trans, delta_angle;
        odomTFSimulation(noise, delta_pose, scan_tf, "odom", "scan");
        if (record_bag == true) {
            sendStampedTranform(scan_tf);
            sendStampedTranform(odom_tf);
            sendStampedTranform(gt_tf);
            if (scan_time_sum > scan_interval) {
                scan_time_sum = 0.0;
                sensor_msgs::LaserScan scan;
                tf::tfMessage gt_tfmsg, scan_tfmsg;
                makeScan(range, angles, scan, "scan", scan_interval / 1e3);
                stampedTransform2TFMsg(gt_tf, gt_tfmsg);
                stampedTransform2TFMsg(scan_tf, scan_tfmsg);
                // bag.write("gt_tf", ros::Time::now(), gt_tfmsg);
                bag.write("scan", ros::Time::now(), scan);
                bag.write("tf", ros::Time::now(), scan_tfmsg);
                scan.header.frame_id = "scan_gt";           // 可视化发布在真值坐标系下,但rosbag录制时在odom对应的scan系下
                scan_pub.publish(scan);

                ros::Time now_stamp = ros::Time::now();
                if (use_recorded_path)
                    ros::Time now_stamp = now_stamp.fromNSec(path_vec[path_counter](0));
                if (use_recorded_path == false) {
                    record_output << now_stamp.toNSec() << " " << tmp_pose(0) << " " << tmp_pose(1) << " " << tmp_pose(2) << std::endl;
                }
            }
            sensor_msgs::Imu imu_msg;
            nav_msgs::Odometry odom;
            double duration = makeImuMsg(imu_trans, "scan_gt", angle, imu_msg, file);
            makePerturbedOdom(noise, delta_pose, odom, duration, "odom", "scan_gt");
            if (odom_time_sum > odom_interval) {
                odom_time_sum = 0.0;
                odom_pub.publish(odom);
                // bag.write("sim_odom", ros::Time::now(), odom);
                // bag.write("sim_imu", ros::Time::now(), imu_msg);
            }
            imu_pub.publish(imu_msg);       // IMU will be published unconditionally
        }
        
        translation.setZero();
        delta_angle = 0.0;
        translation(0) += cos(angle) * speed[0];
        translation(1) += sin(angle) * speed[0];
        translation(0) += sin(angle) * speed[1];
        translation(1) += -cos(angle) * speed[1];
        delta_angle = speed[2];
        angle += delta_angle;
        if (angle > M_PI)
            angle -= 2 * M_PI;
        else if (angle < -M_PI)
            angle += 2 * M_PI;
        if (exit_flag)
            break;
    }
    bag.close();
    if (imu_plot == true) {
        file->close();
        delete file;
    }
    if (use_recorded_path == false) {
        record_output.close();
        printf("Trajectory record completed.\n");
    }
    double mean_time = time_sum / time_cnt;
    printf("Average running time: %.6lf ms, fps: %.6lf hz\n", mean_time, 1000.0 / mean_time);
    cv::destroyAllWindows();
    return 0;
}
