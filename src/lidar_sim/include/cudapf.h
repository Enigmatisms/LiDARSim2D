#pragma once
#include <vector>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

struct Obsp {
    float x = 0.0;
    float y = 0.0;
    Obsp(): x(0.0), y(0.0) {}
    Obsp(float x, float y): x(x), y(y) {}
    Obsp operator-(const Obsp& obs) const {
        return Obsp(x - obs.x, y - obs.y);
    }
};

/**
 * @brief GPU初始化，负责背面剔除（生成一个flag数组，block共享）
 * @note 此前需要使用vector取出所有片段，才能并行执行，片段直接保存在GPU常量内存中
 * @note 本函数每个线程计算一个是否valid的信息（根据法向量），保存到float数组
 * @param segments constant memory片段信息
 * @param flags 共享内存flags, x, y是观测点位置
 */
__device__ void initialize(const float* const segments, const Obsp* const obs, int id, bool* flags);

/**
 * @brief 并行的z buffer, 渲染深度图
 * @note 输入线段的两个端点，进行“片段着色”
 * @param range 是共享内存的range，只有Z buffer全部计算完（并且转换为int） 才可以开始覆盖
 */
__device__ void singleSegZbuffer(
    const Obsp& p1, const Obsp& p2, const Obsp* const ptcls, 
    const int s_id, const int e_id, const double ang_incre, int* range
);

/// 输入原始的segment
__global__ void particleFilter(
    const Obsp* const ptcls,
    const float* const raw_segs,
    const float* const ref, float* weights,
    const int s_id, const int e_id, const double ang_incre,
    int range_num
);