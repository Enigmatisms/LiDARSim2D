#pragma once
#include <vector>
#include "Volume.hpp"

class ParticleFilter {
public:
    ParticleFilter(const cv::Mat& occ, double _angle_incre, int pnum);
    ~ParticleFilter() {}
public:
    void particleInitialize(const cv::Mat& src);

    void particleUpdate(double mx, double my);

    /// @brief 根据权重进行重要性重采样
    void importanceResampler(const std::vector<double>& weights);

    void filtering(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector2d act_obs, cv::Mat& src);

    void visualizeRay(const std::vector<double>& range, const Eigen::Vector2d& obs, cv::Mat& dst) const;

    void visualizeParticles(const std::vector<double>& weights, cv::Mat& dst) const;

    void edgeIntersect(const Edge& eg, const Eigen::Vector2d& obs, std::vector<double>& range);

    /// @brief 模拟的激光器应该具有噪声
    inline void scanPerturb(std::vector<double>& range);

    /// @brief 机器人的控制与实际运动可能不一致，我们三次控制之后进行一次粒子滤波
    void noisedMotion(double& mx, double& my) {
        mx += std::abs(mx) * rng.gaussian(0.5);
        my += std::abs(my) * rng.gaussian(0.5);
    }

    /// @brief 在观测点obs处计算的range与实际的观测计算概率值（用Gauss），此步相当于weight计算
    static double probaComputation(const std::vector<double>& z, const std::vector<double>& exp_obs);
private:
    const cv::Mat& occupancy;
    int ray_num;
    const int point_num;
    const double angle_incre;
    std::vector<Eigen::Vector2d> particles;
    cv::RNG rng;
};