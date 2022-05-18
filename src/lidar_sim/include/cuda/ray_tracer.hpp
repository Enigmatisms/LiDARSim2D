#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "cuda_err_check.hpp"

#define DEPTH_DIV_NUM 360

struct Point {
    float x = 0.f;
    float y = 0.f;
    __device__ Point() : x(0), y(0) {}
    __device__ Point(float x, float y) : x(x), y(y) {}

    __device__ Point operator-(const Point& p) const {
        return Point(x - p.x, y - p.y);
    }

    __device__ float get_angle() const {
        return atan2f(y, x);
    }

    __device__ float norm() const {
        return sqrtf(x * x + y * y);
    }
};

typedef std::vector<Point> Points;

// 如果地图是动态的，则需要根据此函数进行update（覆盖原来的constant mem）
__host__ void updateMap(const float* const host_segs, size_t byte_num);

__host__ void updateSegments(const float* const host_segs, size_t byte_num);

// 预处理模块，进行back culling以及frustum culling
// 由于所有面片, 激光雷达参数等都在constant memory中 故要传入的input不多
// 输出：每个segment四个值（start_id, end_id, distance, normal_angle）以及此segment是否valid（flags）
// 之后的rayTraceKernel使用global memory中的segment四参数进行运算 GPU不允许直接的global memory to constant memory
// TODO: 使用constant memory先写出初版，之后再考虑用texture memory替换
__global__ void preProcess(
    short* sids, short* eids, float* angles, float* dists,
    bool* flags, short ray_num, short seg_num, float amin,
    float ainc, float px, float py, float p_theta
);

__global__ void rayTraceKernel(
    short* const sids, short* const eids, float* const angles, float* const dists, bool* const flags,
    short seg_base_id, short num_segs, short block_seg_num, float* const ranges, float amin, float ainc, float p_theta
);

__global__ void getMininumRangeKernel(const float* const oct_ranges, float* const output, int range_num);

__global__ void sparsifyScan(const float* const denser, float* const sparser);