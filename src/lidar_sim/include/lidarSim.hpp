#pragma once
#include <vector>
#include "Volume.hpp"

class LidarSim {
public:
    LidarSim(Eigen::Vector3d angles);
    ~LidarSim() {}
public:
    /// @brief 模拟激光器扫描
    void scan(
        const std::vector<std::vector<cv::Point>>& obstacles,
        Eigen::Vector2d act_obs, std::vector<double>& range, cv::Mat& src, double angle
    );

    void scanMakeSparse(const std::vector<double>& range, std::vector<double>& sparse, int angle_offset);
private:
    void visualizeRay(const std::vector<double>& range, const Eigen::Vector2d& obs, cv::Mat& dst, double now_ang) const;

    void edgeIntersect(const Edge& eg, const Eigen::Vector2d& obs, std::vector<double>& range) const;
private:
    double sparse_min;
    double sparse_max;
    double sparse_incre;
    double angle_incre;
    cv::RNG rng;
    int full_num;
    int sparse_ray_num;
};