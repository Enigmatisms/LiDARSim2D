#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

struct Vec2 {
    float x;
    float y;
    __host__ __device__ Vec2(float x, float y): x(x), y(y) {}
    __host__ __device__ Vec2() {}
};

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__ Vec3(): x(0.), y(0.), z(0.) {}
    __host__ __device__ Vec3(float x, float y, float z): x(x), y(y), z(z) {}
};

struct Point {
    float x = 0.f;
    float y = 0.f;
    __device__ Point(): x(0), y(0) {}
    __device__ Point(float x, float y): x(x), y(y) {}
    __device__ Point(const Vec2& vec): x(vec.x), y(vec.y) {}
    __device__ Point(const Vec3& vec): x(vec.x), y(vec.y) {}

    __forceinline__ __device__ void perp() {
        float tmp = x;
        x = -y;
        y = tmp;
    }

    __forceinline__ __device__ Point operator-(const Point& p) const {
        return Point(x - p.x, y - p.y);         // Return value optimized?
    }


    __forceinline__ __device__ Point operator+(const Point& p) const {
        return Point(x + p.x, y + p.y);         // Return value optimized?
    }

    __forceinline__ __device__ Point operator*(float val) const { 
        return Point(x * val, y * val);         // Return value optimized?
    }

    __forceinline__ __device__  float get_angle() const {
        return atan2f(y, x);
    }

    __forceinline__ __device__ float dot(const Point& p) const {
        return x * p.x + y * p.y;
    }

    __forceinline__ __device__ float norm() const {
        return sqrtf(x * x + y * y);
    }
};

void copy2ConstMem(const Vec2* const meshes, const char* const nexts, int point_num);

__global__ void backCullPreprocessKernel(
    const float px, const float py, int all_point_num, float* angles, bool* mesh_valid
);              /// logically correct [checked]

__global__ void pointIntersectKernel(
    const float* const rays, const float* const angles, const bool* const mesh_valid, 
    float* outputs, float px, float py, int all_seg_num, int all_ray_num, int ray_boffset, int seg_boffset
);

__global__ void depth2PointKernel(const float* const all_outputs, const float* const ray_angle, size_t seg_block_num, float px, float py, Vec2* const out_pts);

__global__ void simpleDuplicateKernel(const float* const inputs, float* const outputs);
