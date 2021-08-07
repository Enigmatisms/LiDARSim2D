#pragma once
#include <vector>
#include "Edge.hpp"

class ParticleFilter {
public:
    ParticleFilter(double _angle_incre): angle_incre(_angle_incre) {}
    ~ParticleFilter() {}
public:

    /// @brief 在实际观测点act_obs处，发射激光，得到的交点按照角度顺序存储在intersects里面
    /// @note 每个粒子相当于是一个假想的观测点位置，
    /// @param intersects 注意交点是相对于laser坐标系而不是世界坐标系
    void rangeCalculation(const Eigen::Vector2d& act_obs, std::vector<double>& intersects);

    /// @brief 在观测点obs处计算的range与实际的观测计算概率值（用Gauss），此步相当于weight计算
    static double probaComputation(const std::vector<double>& z, const std::vector<double>& exp_obs);

    /// @brief 根据权重进行重要性重采样
    void importanceResampler(
        const std::vector<Eigen::Vector2d>& particles, 
        const std::vector<double>& weights,
        std::vector<Eigen::Vector2d>& resampled
    ) const;

    /// @brief 模拟的激光器应该具有噪声
    static void scanPerturb(std::vector<double>& range);

    /// @brief 机器人的控制与实际运动可能不一致，我们三次控制之后进行一次粒子滤波
    void noisedMotion() const;

    void visualization() const;
private:

private:
    const double angle_incre;
};