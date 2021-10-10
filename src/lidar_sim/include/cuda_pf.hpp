#pragma once
#include "cuda_funcs.h"

#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#define CUDA_CALC_TIME

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line <<
			std::endl;
	exit (1);
}

class CudaPF {
public:
    CudaPF(const cv::Mat& occ, const Eigen::Vector3d& angles, int pnum);
    ~CudaPF() {
        #ifdef CUDA_CALC_TIME
        for (int i = 0; i < 5; i++) {
            double avg_time = time_sum[i] / cnt_sum[i];
            printf("avg_time: %lf ms, avg_fps: %lf\n", avg_time, 1000.0 / avg_time);
        }   
        #endif
        CUDA_CHECK_RETURN(cudaFree(weights));
        CUDA_CHECK_RETURN(cudaFree(ref_range));
    }
public:
    void intialize(const std::vector<std::vector<cv::Point>>& obstacles);

    void particleInitialize(const cv::Mat& src);

    void particleUpdate(double mx, double my);

    /// @brief 根据权重进行重要性重采样
    void importanceResampler(const std::vector<float>& weights);

    void filtering(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector3d act_obs, cv::Mat& src);

    inline void scanPerturb(std::vector<float>& range);

    void noisedMotion(double& mx, double& my) {
        mx += std::abs(mx) * rng.gaussian(0.5);
        my += std::abs(my) * rng.gaussian(0.5);
    }

private:
    const cv::Mat& occupancy;
    int ray_num;
    int seg_num;
    size_t shared_to_allocate;
    const int point_num;
    const double angle_min;
    const double angle_max;
    const double angle_incre;

    // CUDA
    float* weights;
    float* ref_range;
    
    Obsp* cu_pts;
    Obsp* obs;

    std::vector<Eigen::Vector2d> particles;
    cv::RNG rng;
    #ifdef CUDA_CALC_TIME
        std::array<double, 5> time_sum;
        std::array<double, 5> cnt_sum;
    #endif // CUDA_CALC_TIME
};