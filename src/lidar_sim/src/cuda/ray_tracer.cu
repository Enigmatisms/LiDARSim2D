#include <sm_60_atomic_functions.h>
#include "cuda/cuda_err_check.hpp"
#include "cuda/ray_tracer.hpp"

constexpr float _M_2PI = 2.f * M_PIf32;

// 4 * 2048 (for mesh segments)
__constant__ float const_mem[8192];

__host__ void updateSegments(const float* const host_segs, size_t byte_num) {
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(const_mem, host_segs, byte_num, 0, cudaMemcpyHostToDevice));
}

inline __host__ __device__ float goodAngle(float angle) {
    if (angle > M_PIf32) 
        return angle - _M_2PI;
    else if (angle < -M_PIf32)
        return angle + _M_2PI;
    return angle;
}

/// block与thread的设置？由于每次是对面片进行处理，故仍然是分组处理
/// 面片数量 >> 7 为block数量，每个block至多处理128个面片（在少面片时效率不一定特别高）
/// sids, eids, angles, dists的大小为总面片数量
__global__ void preProcess(
    short* sids, short* eids, float* angles, float* dists,
    bool* flags, short ray_num, short seg_num, float amin,
    float ainc, float px, float py, float p_theta
) {
    // no need to use shared memory
	const short tmp_seg_id = ((blockIdx.x << 8) + threadIdx.x);
    if (tmp_seg_id < seg_num) {         // 虽然会有warp divergence，但是应该不影响什么，因为else块没有需要执行的内容
        const short seg_base = tmp_seg_id << 2;
        // printf("Error %d, %d, %x, %x, %x\n", tmp_seg_id, seg_num, uint64_t(sids), uint64_t(eids), uint64_t(dists));
        const Point sp_dir(const_mem[seg_base] - px, const_mem[seg_base + 1] - py), ep_dir(const_mem[seg_base + 2] - px, const_mem[seg_base + 3] - py);
        const float sp_dir_angle = sp_dir.get_angle(), ep_dir_angle = ep_dir.get_angle(), 
            normal_angle = goodAngle((ep_dir - sp_dir).get_angle() + M_PI_2f32);
        const float distance = sp_dir.norm() * cosf(normal_angle - sp_dir_angle);
        dists[tmp_seg_id] = distance;
        angles[tmp_seg_id] = normal_angle;
        const float cur_amin = goodAngle(p_theta + amin);
        const float sp_dangle = (sp_dir_angle - cur_amin), ep_dangle = (ep_dir_angle - cur_amin);
        const short tmp_sid = static_cast<short>(round((sp_dangle + _M_2PI * (sp_dangle < 0)) / ainc)),
            tmp_eid = static_cast<short>(round((ep_dangle + _M_2PI * (ep_dangle < 0)) / ainc));
        // valid id is in range [0, ray_num - 1], all ids are bigger than 0
        flags[tmp_seg_id] = (distance < 0) && ((tmp_eid < ray_num) || (tmp_sid < ray_num));               // back culling + at least either sp or ep is in range
        sids[tmp_seg_id] = tmp_sid;
        eids[tmp_seg_id] = tmp_eid;
    }
}

/**
 * 预处理结束之后，输出: sids, eids, angles, dists, flags
 * 此后分block处理，每个block将与自己相关的信息保存到shared 线程数量为？线程数量是一个比较大的问题，一个block的线程数量不能超过1024，用满不是很好，256左右就已经够了吧
 * 个人不使用分区标号（比较繁琐，warp divergence无法避免），直接遍历所有的valid面片
 * 深度图分区渲染
 */
// 可以尝试采用双分区方法 --- 每个grid的x方向分为8份（面片分8份），y则是角度分区，则最后将输出到8个单线深度图中，最后将8个单线深度图组合在一起
__global__ void rayTraceKernel(
    short* const sids, short* const eids, float* const angles, float* const dists, bool* const flags,
    short seg_base_id, short num_segs, short block_seg_num, float* const ranges, float amin, float ainc, float p_theta
) {
    extern __shared__ float local_segements[]; 
    // local segments大小应该是 4B * (angles + dists) / 8 + DEPTH_DIV_NUM * 4B (深度图分区) + 2B * (sids + eids) / 8 + (1B * len(flags) / 8) + padding
    const short seg_sid = seg_base_id * num_segs, seg_eid = seg_sid + block_seg_num - 1;
    const short tid = threadIdx.x, rimg_id = blockIdx.x * DEPTH_DIV_NUM + tid;
    float* const range_ptr  = (float*)&   local_segements[num_segs << 1];
    short* const id_ptr     = (short*)& local_segements[DEPTH_DIV_NUM + (num_segs << 1)];
    bool* const flag_ptr    = (bool*)&  local_segements[DEPTH_DIV_NUM + (num_segs << 1) + num_segs];   
    // 可能有严重的warp divergence
    for (short i = 0; i < 16; i++) {
        const short local_i = tid + i * DEPTH_DIV_NUM, global_i = seg_sid + local_i, doubled_i = local_i << 1;
        if (global_i > seg_eid) break;     // warp divergence
        flag_ptr[local_i] = flags[global_i];
        if (flag_ptr[local_i] ==  false) continue;
        
        local_segements[doubled_i] = angles[global_i]; 
        local_segements[doubled_i + 1] = dists[global_i]; 
        id_ptr[doubled_i] = sids[global_i];
        id_ptr[doubled_i + 1] = eids[global_i];
    }
    range_ptr[tid] = 1e6;
    __syncthreads();
    // id_ptr 的长度是 2 * num_segs, local_segements(angles dists部分)也是2 * num_segs, flag则是num_segs
    for (short i = 0, ii = 0; i < block_seg_num; i++, ii += 2) {         // traverse all segs
        if (flag_ptr[i] == false) continue;            // 无效首先跳过
        // 奇异判定
        const short start_id = id_ptr[ii], end_id = id_ptr[1 + ii];
        bool singular = (start_id > end_id), less_s = (rimg_id < start_id), more_e = (rimg_id > end_id);
        // 此处的逻辑是：当出现sid > eid时，需要id > end && id < start, 而反之则只需要 id > end || id < start. 这样写为了防止更多的warp divergence. 使用卡诺图简化了计算 原型见89a0070d
        if ((less_s && more_e) || ((less_s || more_e) && !singular)) continue;
        float local_range = local_segements[ii + 1] / cosf(local_segements[ii] - (amin + p_theta + ainc * float(rimg_id)));
        // if (local_range < -1e-3 || local_range > 1e4) {
            // printf("Error: %f, %f, %f\n", local_segements[ii + 1], local_segements[ii], (amin + p_theta + ainc * float(rimg_id)));
        // }
        range_ptr[tid] = std::min(range_ptr[tid], local_range);
    }
    // 处理结束，将shared memory复制到global memory
    __syncthreads();
    ranges[rimg_id] = range_ptr[tid];
}

// 将8个单线深度图合并为一个
__global__ void getMininumRangeKernel(const float* const oct_ranges, float* const output, int range_num) {
    const int range_base = DEPTH_DIV_NUM * blockIdx.x + threadIdx.x;
    float min_range = 1e9;
    for (int i = 0, tmp_base = 0; i < 8; i++, tmp_base += range_num) {
        min_range = std::min(min_range, oct_ranges[range_base + tmp_base]);
    }
    output[range_base] = min_range;
}

__global__ void sparsifyScan(const float* const denser, float* const sparser) {
    // ray_per_block will be 120
    const int output_base = blockDim.x * blockIdx.x + threadIdx.x, input_base = output_base * 3;
    float r_sum = 0.0f;
    int avg_cnt = 0;
    for (int i = 0; i < 3; i++) {
        float r = denser[input_base + i];
        bool range_valid = (r < 1e5);
        r_sum += r * range_valid;
        avg_cnt += range_valid;
    }
    if (avg_cnt > 0) {
        sparser[output_base] = (r_sum / float(avg_cnt));
    } else {
        sparser[output_base] = 1e6;
    }
}
