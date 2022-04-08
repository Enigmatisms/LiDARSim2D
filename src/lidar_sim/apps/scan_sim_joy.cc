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
    speed[1] = -reinterpret_cast<float &>(joyc->lateral_speed) * trans_speed_amp;
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
    std::string scan_topic = nh.param<std::string>("/scan_joy/scan_topic", "scan");
    std::string odom_topic = nh.param<std::string>("/scan_joy/odom_topic", "odom");
    std::string imu_topic = nh.param<std::string>("/scan_joy/odom_topic", "imu");
    std::string bag_name = nh.param<std::string>("/scan_joy/bag_name", "standard");
    trans_speed_amp = nh.param<double>("/scan_joy/trans_speed_amp", 3.0);
    rot_vel_amp = nh.param<double>("/scan_joy/rot_vel_amp", 1.5);
    double init_x = nh.param<double>("/scan_joy/init_x", 367.0);
    double init_y = nh.param<double>("/scan_joy/init_y", 769.0);
    double angle_min = nh.param<double>("/scan_joy/angle_min", -M_PI / 2);
    double angle_max = nh.param<double>("/scan_joy/angle_max", M_PI / 2);
    double angle_incre = nh.param<double>("/scan_joy/angle_incre", M_PI / 1800.0);
    double display_rate = nh.param<double>("/scan_joy/display_rate", 20.0);

    double translation_noise = nh.param<double>("/scan_joy/translation_noise", 0.08);
    double rotation_noise = nh.param<double>("/scan_joy/rotation_noise", 0.01);
    double trans_vel_noise = nh.param<double>("/scan_joy/trans_vel_noise", 0.01);
    double rot_vel_noise = nh.param<double>("/scan_joy/rot_vel_noise", 0.01);

    double lidar_noise = nh.param<double>("/scan_joy/lidar_noise", 0.02);
    const double pix_resolution = nh.param<double>("/scan_joy/pix_resolution", 0.02);
    int lidar_multiple = nh.param<int>("/scan_joy/lidar_multiple", 2);
    bool skip_selection = nh.param<bool>("/scan_joy/skip_selection", false);
    bool imu_plot = nh.param<bool>("/scan_joy/imu_plot", false);
    bool use_recorded_path = nh.param<bool>("/scan_joy/use_recorded_path", false);
    bool bag_imu = nh.param<bool>("/scan_joy/bag_imu", false);
    bool bag_odom = nh.param<bool>("/scan_joy/bag_odom", false);
    mouse_ctrl = nh.param<bool>("/scan_joy/enable_mouse_ctrl", false);
    K_P = nh.param<double>("/scan_joy/kp", 0.2);
    K_I = nh.param<double>("/scan_joy/ki", 0.0001);
    K_D = nh.param<double>("/scan_joy/kd", 0.01);


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
    int scan_cnt = 0;
    Eigen::Vector3d angles(angle_min, angle_max, angle_incre);
    LidarSim ls(angles, lidar_noise);
    std::vector<Eigen::Vector3d> gtt;       // ground truth tragectory
    rosbag::Bag bag(pack_path + "/../../bags/" + bag_name + ".bag", rosbag::bagmode::Write);
    nav_msgs::Odometry odom;
    TicToc timer, inner_timer;
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
    inner_timer.tic();
    ros::Rate rate(display_rate);
    translation.setZero();
    delta_angle = 0.0;
    while (ros::ok()) {
        cv::Point start(obs.x(), obs.y()), end(obs.x() + 20 * cos(angle), obs.y() + 20 * sin(angle));
        cv::arrowedLine(src, start, end, cv::Scalar(255, 0, 0), 2);
        cv::imshow("disp", src);
        cv::waitKey(1);
        char key = status;
        controlFlow();
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
        plotSpeedInfo(src, trans_speed_amp, act_speed_x);
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
        bool running = (abs(speed[0]) > 1e-5) | (abs(speed[1]) > 1e-5);
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
        double act_trans_x = act_speed_x, act_trans_y = act_speed_y;            // act_trans(相对) 与 translation(绝对) 是完全不同的概念
        if (collided) {
            act_trans_x = 0.0;
            act_trans_y = 0.0;
        }
        Eigen::Vector2d imu_trans(act_trans_x * pix_resolution, act_trans_y * pix_resolution);
        Eigen::Vector3d delta_pose;
        delta_pose << imu_trans, delta_angle;
        odomTFSimulation(noise, delta_pose, odom_tf, odom_topic, scan_topic);
        sensor_msgs::Imu imu_msg;
        nav_msgs::Odometry odom;
        double duration = makeImuMsg(imu_trans, scan_topic, angle, imu_msg, file);
        makePerturbedOdom(noise, init_obs * pix_resolution, delta_pose, odom, init_angle, duration, odom_topic, scan_topic);
        // odom_tf.
        sendStampedTranform(gt_tf);
        tf::StampedTransform odom2map = getOdom2MapTF(gt_tf, odom_tf, init_obs * pix_resolution, init_angle);
        tf::Vector3 odom_translation = odom_tf.getOrigin();
        printf("%lf, %lf, %lf, %lf, %lf\n", delta_pose.x(), delta_pose.y(), delta_pose.z(), odom_translation.x(), odom_translation.y());
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
        translation(0) += cos(angle) * speed[0];
        translation(1) += sin(angle) * speed[0];
        translation(0) += -sin(angle) * speed[1];
        translation(1) += cos(angle) * speed[1];
        delta_angle = speed[2];
        angle += delta_angle;
        if (angle > M_PI)
            angle -= 2 * M_PI;
        else if (angle < -M_PI)
            angle += 2 * M_PI;
        if (exit_flag)
            break;
        rate.sleep();
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
    cv::destroyAllWindows();
    return 0;
}
