#pragma once
#include <iostream>
#include "ray_tracer.hpp"

typedef std::vector<Eigen::Vector2d> Mesh;
typedef std::vector<Mesh> Meshes;


__host__ void intializeFixed(int num_ray);
__host__ void deallocateDevice();
__host__ void deallocateFixed();

__host__ void unwrapMeshes(const Meshes& meshes, bool initialized = false);

__host__ double rayTraceRenderCpp(const Eigen::Vector3d& lidar_param, const Eigen::Vector2d& pose, float angle, int ray_num, std::vector<float>& range);
