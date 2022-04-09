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
#include <geometry_msgs/Quaternion.h>

extern double K_P;
extern double K_I;
extern double K_D;

template<typename T>
std::string toString(T data) {
    std::stringstream ss;
    ss << data;
    std::string s;
    ss >> s;
    return s;
}

inline double goodAngle(double angle) {
    if (angle > M_PI)
        return angle - 2 * M_PI;
    else if (angle < -M_PI)
        return angle + 2 * M_PI;
    return angle;
}

template <typename T>
inline double quaterion2Angle(T qt) {
    return goodAngle(atan2(qt.z(), qt.w()) * 2.0);
}

template <>
inline double quaterion2Angle<geometry_msgs::Quaternion>(geometry_msgs::Quaternion qt) {
    return goodAngle(atan2(qt.z, qt.w) * 2.0);
}

void initializeReader(std::ifstream& recorded_path, std::vector<std::array<double, 8>>& path_vec, Eigen::Vector2d& init_obs, double& init_angle);
void initializeWriter(std::ofstream& record_output, const Eigen::Vector2d& init_obs, double init_angle);

// scan带有漂移的odometry transform
void odomTFSimulation(
    const Eigen::Vector4d& noise_level, Eigen::Vector3d delta_p, tf::StampedTransform& tf, 
    Eigen::Vector3d& noise, std::string frame_id, std::string child_id
);

tf::StampedTransform getOdom2MapTF(const tf::StampedTransform& scan2map, const tf::StampedTransform& scan2odom, const Eigen::Vector2d& init_obs, double init_angle);

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
    const Eigen::Vector4d& noise_level, const Eigen::Vector2d& init_pos, Eigen::Vector3d delta_p, Eigen::Vector3d noise,
    nav_msgs::Odometry& odom, double init_angle, double duration, std::string frame_id, std::string child_id
);

void sendTransform(Eigen::Vector3d p, std::string frame_id, std::string child_frame_id);

void sendStampedTranform(const tf::StampedTransform& _tf);

double pidAngle(const Eigen::Vector2d& orient, const Eigen::Vector2d& obs, double now);

void makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    double duration,
    sensor_msgs::Imu& msg,
    Eigen::Vector2d vel_var,
    Eigen::Vector2d ang_var
);

double makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    double duration,
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
    double toc() const {
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0;
    }
};