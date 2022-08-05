#include <vector>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "cuda/cuda_err_check.hpp"
#include "cuda/cast_kernel.hpp"

#define PREPROCESS_BLOCK 4

float* point_angles = nullptr, *sorted_angles = nullptr;
bool* next_valid = nullptr;
int all_point_num = 0;              // set in memAllocator

void deallocatePoints() {
    CUDA_CHECK_RETURN(cudaFree(point_angles));
    CUDA_CHECK_RETURN(cudaFree(sorted_angles));
    CUDA_CHECK_RETURN(cudaFree(next_valid));
}

void updatePointInfo(const Vec2* const meshes, const char* const nexts, int point_num, bool initialized) {
    copy2ConstMem(meshes, nexts, point_num);
    if (initialized == true)
        deallocatePoints();
    CUDA_CHECK_RETURN(cudaMalloc((void**) &point_angles, sizeof(float) * point_num));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &sorted_angles, sizeof(float) * point_num));
    CUDA_CHECK_RETURN(cudaMalloc((void**) &next_valid, sizeof(bool) * point_num));
    all_point_num = point_num;
}

__global__ void printTest(const float* const sorted_angles, const float* const actual_rays, int invalid_bound, int actual_ray_num) {
    printf("Sorted angles: ");
    for (int i = 0; i < invalid_bound; i++) {
        printf("%f, ", sorted_angles[i]);
    }
    printf("\n");
    printf("Doubled angles: ");
    for (int i = 0; i < actual_ray_num; i++) {
        printf("%f, ", actual_rays[i]);
    }
    printf("\n");
}

/// *
void shadowCasting(const Vec3& pose, std::vector<Vec2>& host_output) {
    const int thread_per_block = static_cast<int>(std::ceil(static_cast<float>(all_point_num) / PREPROCESS_BLOCK));
    backCullPreprocessKernel<<< PREPROCESS_BLOCK, thread_per_block >>> (pose.x, pose.y, all_point_num, point_angles, next_valid);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // sorting rays
    CUDA_CHECK_RETURN(cudaMemcpy(sorted_angles, point_angles, all_point_num * sizeof(float), cudaMemcpyDeviceToDevice));
    thrust::sort(thrust::device, sorted_angles, sorted_angles + all_point_num, thrust::less<float>());
    const int invalid_bound = thrust::lower_bound(thrust::device, sorted_angles, sorted_angles + all_point_num, 1e2, thrust::less<float>()) - sorted_angles;
    /// duplicate valid rays
    float* actual_rays = nullptr, *output_depth = nullptr;
    Vec2* out_pts = nullptr;
    const int actual_ray_num = invalid_bound << 1;              // double the number of valid rays (left & right extensions)
    const size_t ray_size = sizeof(float) * actual_ray_num;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &actual_rays, sizeof(float) * actual_ray_num));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &out_pts, sizeof(Vec2) * actual_ray_num));
    size_t ray_block_num = static_cast<size_t>(std::ceil(static_cast<float>(actual_ray_num) / 16));
    size_t seg_block_num = static_cast<size_t>(std::ceil(static_cast<float>(all_point_num) / 64));
    simpleDuplicateKernel<<< 1, invalid_bound >>> (sorted_angles, actual_rays);
    CUDA_CHECK_RETURN(cudaMalloc((void **) &output_depth, ray_size * seg_block_num));
    
    // printTest<<< 1, 1>>>(sorted_angles, actual_rays, invalid_bound, actual_ray_num);
    /// get ray - mesh segment intersections. Notice that point_num (all_point_num) equals number of segment
    const size_t shared_memory_size = (sizeof(int) << 4);
    cudaStream_t streams[4];
    for (short i = 0; i < 4; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    // printf("ray block num: %d, seg block num: %d, ray num: %d, seg num: %d\n", ray_block_num, seg_block_num, actual_ray_num, all_point_num);
    for (size_t i = 0, stream_idx = 0; i < seg_block_num; i += 4) {				// 面片
        for (size_t j = 0; j < ray_block_num; j += 4, stream_idx++) {			// 光线
            // pose 在本处是const Vec3&, 在进入kernel时会发生复制，可以吗？
            dim3 dimGrid(4, 4);
            dim3 dimBlock(16, 64);
            pointIntersectKernel<<<dimGrid, dimBlock, shared_memory_size, streams[stream_idx % 4]>>>(
                actual_rays, point_angles, next_valid, output_depth, pose.x, pose.y, all_point_num, actual_ray_num, j, i
            );
        }
    }
    /// output, cleaning up
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    depth2PointKernel<<<1, actual_ray_num>>>(output_depth, actual_rays, seg_block_num, pose.x, pose.y, out_pts);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    /// *
    host_output.resize(actual_ray_num);
    CUDA_CHECK_RETURN(cudaMemcpy(host_output.data(), out_pts, sizeof(Vec2) * actual_ray_num, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; i++)
        cudaStreamDestroy(streams[i]);
    CUDA_CHECK_RETURN(cudaFree(out_pts));
    CUDA_CHECK_RETURN(cudaFree(actual_rays));
    CUDA_CHECK_RETURN(cudaFree(output_depth));
}