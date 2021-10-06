/**
 * @file scanUtils.hpp
 * @author Enigmatisms
 * @brief Utilities for 2d scan simulator
 * @date 2021-08-27
 * @copyright Copyright (c) 2021
 */

#pragma once
#include <chrono>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>

extern double K_P;
extern double K_I;
extern double K_D;
extern Eigen::Vector3d __pose__;

void stampedTransform2TFMsg(const tf::StampedTransform& transf, tf::tfMessage& msg);

/// @brief create transform - tf messages
void makeTransform(Eigen::Vector3d p, std::string frame_id, std::string child_frame_id, tf::StampedTransform& transf);

/// @brief 2D LiDAR scan messages
void makeScan(
    const std::vector<double>& range, const Eigen::Vector3d& angles,
    sensor_msgs::LaserScan& scan, std::string frame_id, double scan_time
);

/// @brief 2D Odometry with noise
/// @param noise_level the first elem of this var is the noise level in translation, the second is for rotation
void makePerturbedOdom(
    Eigen::Vector3d delta_p, nav_msgs::Odometry& odom, 
    Eigen::Vector2d noise_level, std::string frame_id, std::string child_id
);

void sendTransform(Eigen::Vector3d p, std::string frame_id, std::string child_frame_id);

void sendStampedTranform(const tf::StampedTransform& _tf);

double pidAngle(const Eigen::Vector2d& orient, const Eigen::Vector2d& obs, double now);

void makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    sensor_msgs::Imu& msg,
    Eigen::Vector2d vel_var = Eigen::Vector2d::Zero(),
    Eigen::Vector2d ang_var = Eigen::Vector2d::Zero()
);

class TicToc {
private:
    std::chrono::_V2::system_clock::time_point start;
public:
    void tic() {
        start = std::chrono::system_clock::now();
    }

    // return in milliseconds
    double toc() {
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0;
    }
};