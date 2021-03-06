/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <iostream>

struct Obsp {
    float x = 0.0;
    float y = 0.0;
    float a = 0.0;
    __host__ __device__ Obsp(): x(0.0), y(0.0), a(0.0) {}
    __host__ __device__ Obsp(float x, float y, float a): x(x), y(y), a(a) {}
};

struct Vec2f {
    float x = 0.0;
    float y = 0.0;
    __device__ Vec2f(): x(0.0), y(0.0) {}
    __device__ Vec2f(float x, float y): x(x), y(y) {}
    __device__ __forceinline__ Vec2f operator+(const Vec2f& v) const {
        return Vec2f(x + v.x, y + v.y);
    }
    __device__ __forceinline__ Vec2f operator-(const Vec2f& v) const {
        return Vec2f(x - v.x, y - v.y);
    }
    __device__ __forceinline__ Vec2f operator*(const float v) const {
        return Vec2f(x * v, y * v);
    }
    __device__ __forceinline__ float norm() const {
        return sqrtf(x * x + y * y);
    }
};

__host__ void copyRawSegs(const float* const host_segs, size_t byte_num);

/**
 * @brief GPU初始化，负责背面剔除（生成一个flag数组，block共享）
 * @note 此前需要使用vector取出所有片段，才能并行执行，片段直接保存在GPU常量内存中
 * @note 本函数每个线程计算一个是否valid的信息（根据法向量），保存到float数组
 * @param segments constant memory片段信息
 * @param flags 共享内存flags, x, y是观测点位置
 */
__device__ void initialize(const float* const segments, const Vec2f* const obs, short id, bool* flags);

/**
 * @brief 并行的z buffer, 渲染深度图
 * @note 输入线段的两个端点，进行“片段着色”
 * @param range 是共享内存的range，只有Z buffer全部计算完（并且转换为int） 才可以开始覆盖
 */
__device__ void singleSegZbuffer(
    const Vec2f& p1, const Vec2f& p2, const Obsp* const ptcls, 
    const short s_id, const short e_id, const int range_num, const float ang_incre, int* range
);

/// 输入原始的segment
__global__ void particleFilter(
    const Obsp* const ptcls,
    const float* const ref, float* weights,
    const float ang_min, const float ang_incre, const int range_num, 
    const int full_rnum, const int offset, const bool single_flag = false
);
