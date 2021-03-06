/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#pragma once
#include "cuda_funcs.h"
#include <vector>
#include <fstream>
#include <opencv2/core.hpp>
#include "cuda_err_check.hpp"
#define CUDA_CALC_TIME
#define SAVE_RANGE_FILE

class CudaPF {
public:
    CudaPF(const cv::Mat& occ, const Eigen::Vector3d& angles, int pnum, std::string path = "");
    ~CudaPF() {
        #ifdef CUDA_CALC_TIME
        for (int i = 0; i < 5; i++) {
            double avg_time = time_sum[i] / cnt_sum[i];
            printf("avg_time: %lf ms, avg_fps: %lf\n", avg_time, 1000.0 / avg_time);
        }   
        #endif
        CUDA_CHECK_RETURN(cudaFree(weights));
        CUDA_CHECK_RETURN(cudaFree(ref_range));
        CUDA_CHECK_RETURN(cudaFreeHost(particles));
        #ifdef SAVE_RANGE_FILE
        file.close();
        #endif
    }
public:
    void intialize(const std::vector<std::vector<cv::Point>>& obstacles);

    void particleInitialize(const cv::Mat& src, Eigen::Vector3d act_obs = Eigen::Vector3d::Zero());

    void particleUpdate(const Eigen::Vector3d& act_obs, double mx, double my, double angle);

    void singleDebugDemo(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector3d act_obs, cv::Mat& src);

    void filtering(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector3d act_obs, cv::Mat& src);
private:
    /// @brief 根据权重进行重要性重采样
    void importanceResampler(const std::vector<float>& weights);
    
    void visualizeParticles(const std::vector<float>& weights, cv::Mat& dst) const;

    inline void scanPerturb(std::vector<float>& range);

    void noisedMotion(double& mx, double& my, double& angle) {
        mx += std::abs(mx) * rng.gaussian(1.5);
        my += std::abs(my) * rng.gaussian(1.5);
        angle += std::abs(angle) * rng.gaussian(0.06);
    }
private:
    const cv::Mat& occupancy;
    int ray_num;
    int full_ray_num;

    int seg_num;
    size_t shared_to_allocate;
    const int point_num;
    const int cascade_num;
    const double angle_min;
    const double angle_max;
    const double angle_incre;

    // CUDA
    float* weights;
    float* ref_range;
    
    Obsp* cu_pts;
    Obsp* obs;

    Obsp* particles;
    std::vector<float> host_seg;
    std::vector<float> weight_vec;
    cv::RNG rng;
    #ifdef CUDA_CALC_TIME
        std::array<double, 5> time_sum;
        std::array<double, 5> cnt_sum;
    #endif // CUDA_CALC_TIME
    #ifdef SAVE_RANGE_FILE
        std::ofstream file;
    #endif
};
