#include <cstdio>
#include "cuda/cast_kernel.hpp"
#include "cuda/cuda_err_check.hpp"
#include <sm_60_atomic_functions.h>

__constant__ Vec2 all_points[4096];
__constant__ char next_ids[4096];

constexpr int BIG_FLOAT = 0xc6fffe00;      // when convert to float directly, this is equal to 32767.0

void copy2ConstMem(const Vec2* const meshes, const char* const nexts, int point_num) {
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(all_points, meshes, sizeof(Vec2) * point_num, 0, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(next_ids, nexts, sizeof(char) * point_num, 0, cudaMemcpyHostToDevice));
}

__device__ __forceinline__ int floatToOrderedInt( const float floatVal ) {
    const int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( const int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}

__global__ void backCullPreprocessKernel(
    const float px, const float py, int all_point_num, float* angles, bool* mesh_valid
) {
    const int point_base = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_base < all_point_num) {              // for the last block, thread might be less than act_point_num
        // printf("p base: %d / %d\n", point_base, all_point_num);
        const int next_id = static_cast<int>(next_ids[point_base]);
        const int need_forward = next_id > 0, need_reverse = next_id < 0;
        const int next_pt_id = point_base + next_id * need_reverse + 1 - need_reverse;
        const int prev_pt_id = point_base + next_id * need_forward - 1 + need_forward;
        // if (next_pt_id < 0 || next_pt_id >= all_point_num || prev_pt_id < 0 || prev_pt_id >= all_point_num) {
        //     printf("next_pt: %d, prev_pt: %d, all_pt: %d\n", next_pt_id, prev_pt_id, all_point_num);
        // }
        Point cur_pt = Point(all_points[point_base]);
        const Point cur_vec = cur_pt - Point(px, py);
        Point prev_vec = cur_pt - Point(all_points[prev_pt_id]), next_vec = Point(all_points[next_pt_id]) - cur_pt;
        prev_vec.perp();
        next_vec.perp();
        const bool next_valid = (next_vec.dot(cur_vec) < 0.), point_valid = (next_valid || (prev_vec.dot(cur_vec) < 0.));
        angles[point_base] = (1 - point_valid) * 1e3 + point_valid * cur_vec.get_angle();
        mesh_valid[point_base] = next_valid;
        // printf("mesh[%d / %d] (pt = %f, %f, next = %f, %f, cur_vec = %f, %f) is (%d)\n", point_base, next_id, cur_pt.x, cur_pt.y, next_vec.x, next_vec.y, cur_vec.x, cur_vec.y, next_valid);
    }
    __syncthreads();
}

__global__ void simpleDuplicateKernel(
    const float* const inputs, float* const outputs
) {
    const float input_angle = inputs[threadIdx.x];
    outputs[threadIdx.x << 1] = input_angle - 2e-5;
    outputs[1 + (threadIdx.x << 1)] = input_angle + 2e-5;
}

__global__ void pointIntersectKernel(
    const float* const rays, const float* const angles, const bool* const mesh_valid, 
    float* outputs, float px, float py, int all_seg_num, int all_ray_num, int ray_boffset, int seg_boffset
) {
    // blockIdx.x 是 本次处理的面片block id, y是光线块id，threadIdx.x 是光线内部偏移，y是面片id
    extern __shared__ int mini_depth[];     // ordered int for depth atomic comp
    // printf("intersect kernel is called.\n");
    if (threadIdx.y == 0) {                 // shared memory initialization
        mini_depth[threadIdx.x] = BIG_FLOAT;
    }
    __syncthreads();
    const int seg_base = seg_boffset + blockIdx.x, ray_base = ray_boffset + blockIdx.y;
    const int seg_id = seg_base * blockDim.y + threadIdx.y;
    const int ray_id = ray_base * blockDim.x + threadIdx.x;
    
    if (seg_id < all_seg_num && mesh_valid[seg_id] == true && ray_id < all_ray_num) {
        const float ray_angle = rays[ray_id];
        const int next_id = static_cast<int>(next_ids[seg_id]), next_id_neg = next_id < 0;
        const int next_angle_id = seg_id + 1 - next_id_neg + next_id * next_id_neg;
        const float cur_angle = angles[seg_id], next_angle = angles[next_angle_id];
        const bool not_singular = (cur_angle < next_angle);
        const bool in_range = (ray_angle > cur_angle) && (ray_angle < next_angle);
        const bool in_range_singular = (ray_angle > cur_angle) || (ray_angle < next_angle);
        if ((in_range && not_singular) || (in_range_singular && !not_singular)) {
            const Point dir_vec(cosf(ray_angle), sinf(ray_angle)), this_p(all_points[seg_id]), obs_p(px - this_p.x, py - this_p.y);
            Point p = Point(all_points[next_angle_id]) - this_p;
            p.perp();
            // if (seg_id == 32) {
            //     printf("seg[33]\n");
            // }
            float depth = -p.dot(obs_p) / p.dot(dir_vec);               // 需要求解此depth的最小值
            // const int output_id = ((seg_boffset << 1) + blockIdx.x) * blockDim.x + threadIdx.x;
            int ordered_int = floatToOrderedInt(depth);
            atomicMin(mini_depth + threadIdx.x, ordered_int);
        }
    }
    __syncthreads();
    // if (threadIdx.x == 0 && seg_id == 32) {
    //     printf("%d, %d\n", all_seg_num, mesh_valid[seg_id]);
    // }
    if (threadIdx.y == 0 && ray_id < all_ray_num) {
        const int output_id = seg_base * all_ray_num + ray_id;
        outputs[output_id] = orderedIntToFloat(mini_depth[threadIdx.x]);
    }
    __syncthreads();
}

__global__ void depth2PointKernel(const float* const all_outputs, const float* const ray_angle, size_t seg_block_num, float px, float py, Vec2* const out_pts) {
    const int ray_num = blockDim.x, ray_id = threadIdx.x;
    float min_depth = 32767.0;
    for (int i = 0; i < seg_block_num; i++) {
        min_depth = min(min_depth, all_outputs[i * ray_num + ray_id]);
    }
    // printf("Min depth of %d = %f (angle: %f)\n", threadIdx.x, min_depth, ray_angle[ray_id]);
    const float angle = ray_angle[ray_id];
    const Point op = Point(cosf(angle), sinf(angle)) * min_depth + Point(px, py);
    Vec2& output = out_pts[ray_id];
    output.x = op.x;
    output.y = op.y;
}
