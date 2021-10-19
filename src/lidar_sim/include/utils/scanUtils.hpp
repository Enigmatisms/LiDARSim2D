/**
 * @file scanUtils.hpp
 * @author (Enigmatisms:https://github.com/Enigmatisms) @copyright Qianyue He
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

// scan带有漂移的odometry transform
void odomTFSimulation(
    const Eigen::Vector4d& noise_level, Eigen::Vector3d delta_p, tf::StampedTransform& tf, 
    std::string frame_id, std::string child_id
);

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
    const Eigen::Vector4d& noise_level, Eigen::Vector3d delta_p, nav_msgs::Odometry& odom, 
    double duration, std::string frame_id, std::string child_id
);

void sendTransform(Eigen::Vector3d p, std::string frame_id, std::string child_frame_id);

void sendStampedTranform(const tf::StampedTransform& _tf);

double pidAngle(const Eigen::Vector2d& orient, const Eigen::Vector2d& obs, double now);

std::pair<double, double> makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    sensor_msgs::Imu& msg,
    Eigen::Vector2d vel_var,
    Eigen::Vector2d ang_var
);

double makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    sensor_msgs::Imu& msg,
    std::ofstream* file = nullptr
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