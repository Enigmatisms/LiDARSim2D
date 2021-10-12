#include "cuda_funcs.h"
#include "cuda_err_check.hpp"
#include <sm_60_atomic_functions.h>

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0
constexpr float _M_2PI = 2 * M_PI;
__constant__ float raw_segs[2048];

__host__ void copyRawSegs(const float* const host_segs, size_t byte_num) {
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(raw_segs, host_segs, byte_num, 0, cudaMemcpyHostToDevice));
}

__device__ __forceinline__ int floatToOrderedInt( const float floatVal ) {
    const int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( const int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}

__device__ void initialize(const float* const segments, const Eigen::Vector2f* const obs, int id, bool* flags) {
    const float* ptr = segments + (id << 2);
    const Eigen::Vector2f pt1(ptr[0], ptr[1]), pt2(ptr[2], ptr[3]);
    const Eigen::Vector2f norm(pt1.y() - pt2.y(), pt2.x() - pt1.x());
    const Eigen::Vector2f ctr_vec = (pt1 + pt2) * 0.5 - Eigen::Vector2f(obs->x(), obs->y());
    if (ctr_vec.x() * norm.x() + ctr_vec.y() * norm.y() > 0.0) {
        flags[id] = false;
    } else {
        flags[id] = true;
    }
}

template<bool singl>
__device__ bool isIdInRange(const int range_sid, const int range_eid, const int id) {
    if (singl == true)
        return (id >= range_sid) || (id <= range_eid);
    else
        return (id >= range_sid) && (id <= range_eid);
}

template<bool singl>
__device__ int getRangeOffset(const int range_sid, const int range_eid, const int id, const int range_num) {
    if (singl == true)
        if (id <= range_eid)
            return max(range_num - range_eid + id - 1, 0);
    return min(id - range_sid, range_num - 1);
}

__device__ float getRange(const Eigen::Vector2f& p1, const Eigen::Vector2f& vec_line, const Eigen::Vector2f& obs_pt, const float angle) {
    const Eigen::Vector2f vec(cos(angle), sin(angle));
    Eigen::Matrix2f A;
    A << vec_line(0), -vec(0), vec_line(1), -vec(1);
    const float b1 = -vec(1) * obs_pt(0) + vec(0) * obs_pt(1);
    const float b2 = -vec_line(1) * p1(0) + vec_line(0) * p1(1);
    const Eigen::Vector2f b(b1, b2);
    const float det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    if (abs(det) < 1e-5) {
        return (p1 - obs_pt).norm();
    }
    A /= det;
    return (A * b - obs_pt).norm();
}

__device__ void singleSegZbuffer(
    const Eigen::Vector2f& p1, const Eigen::Vector2f& p2, const Obsp* const ptcls,
    const int s_id, const int e_id, const int range_num,
    const float ang_incre, int* range
) {
    const Eigen::Vector2f obs(ptcls->x, ptcls->y);
    const Eigen::Vector2f ray1 = p1 - obs, ray2 = p2 - obs, vec_line = p2 - p1;
    const float sang = atan2f(ray1(1), ray1(0)), eang = atan2f(ray2(1), ray2(0));
    const int id_s = static_cast<int>(ceilf((sang + M_PI) / ang_incre)),
        id_e = static_cast<int>(floorf((eang + M_PI) / ang_incre));
    if (id_s == id_e + 1) {
        return;
    }
    const int max_ray_num = roundf(_M_2PI / ang_incre);
    bool not_int_range = false;
    const bool range_singl = (s_id > e_id), edge_singl = (id_s > id_e);
    if (range_singl) {          // 深度图范围角度奇异
        if (edge_singl == false) {      // 被投影边角度不奇异
            if (id_e < s_id && id_s > e_id) {
                not_int_range = true;
            }
        }                       // 奇异必定有重合部分
    } else {
        if (edge_singl) {      // 被投影边角度奇异
            if (id_e > e_id && id_s < s_id) {
                not_int_range = true;
            }
        } else {
            if (id_s > e_id || id_e < s_id) {
                not_int_range = true;
            }
        }
    }
    if (not_int_range == false) {   // 不进行动态并行，线程不够用，一个SM才2048个线程
        if (edge_singl) {
            for (int i = id_s; i < max_ray_num; i++) {
                if (range_singl) {       // 超出range范围的不计算
                    if (isIdInRange<true>(s_id, e_id, i) == false) {
                        continue;
                    }
                } else {
                    if (isIdInRange<false>(s_id, e_id, i) == false) {
                        continue;
                    }
                }
                const float angle = ang_incre * static_cast<float>(i) - M_PI;
                const float rval = getRange(p1, vec_line, obs, angle);
                const int range_int  = floatToOrderedInt(rval);
                int offset = 0;
                if (range_singl) {
                    offset = getRangeOffset<true>(s_id, e_id, i, range_num);
                } else {
                    offset = getRangeOffset<false>(s_id, e_id, i, range_num);
                }
                atomicMin(range + offset, range_int);         // 原子压入
            }
            for (int i = 0; i <= id_e; i++) {
                if (range_singl) {       // 超出range范围的不计算
                    if (isIdInRange<true>(s_id, e_id, i) == false) {
                        continue;
                    }
                } else {
                    if (isIdInRange<false>(s_id, e_id, i) == false) {
                        continue;
                    }
                }
                const float angle = ang_incre * static_cast<float>(i) - M_PI;
                const float rval = getRange(p1, vec_line, obs, angle);
                const int range_int  = floatToOrderedInt(rval);
                int offset = 0;
                if (range_singl) {
                    offset = getRangeOffset<true>(s_id, e_id, i, range_num);
                } else {
                    offset = getRangeOffset<false>(s_id, e_id, i, range_num);
                }
                atomicMin(range + offset, range_int);         // 原子压入
            }
        } else {
            for (int i = id_s; i <= id_e; i++) {
                if (range_singl) {       // 超出range范围的不计算
                    if (isIdInRange<true>(s_id, e_id, i) == false) {
                        continue;
                    }
                } else {
                    if (isIdInRange<false>(s_id, e_id, i) == false) {
                        continue;
                    }
                }
                const float angle = ang_incre * static_cast<float>(i) - M_PI;
                const float rval = getRange(p1, vec_line, obs, angle);
                const int range_int  = floatToOrderedInt(rval);
                int offset = 0;
                if (range_singl) {
                    offset = getRangeOffset<true>(s_id, e_id, i, range_num);
                } else {
                    offset = getRangeOffset<false>(s_id, e_id, i, range_num);
                }
                atomicMin(range + offset, range_int);         // 原子压入
            }
        }
    }
}

/// 共享内存需要用在flags / range上
__global__ void particleFilter(
    const Obsp* const ptcls,
    // const float* const raw_segs,
    const float* const ref, float* weights,
    const float ang_min, const float ang_incre, const int range_num, 
    const int full_rnum, const bool single_flag
) {
    extern __shared__ int range[];          //...一个数据类型分为了两个不同意义以及类型的块
    bool* flags = (bool*)(&range[range_num]);
    const int pid = blockIdx.x, sid = threadIdx.x;
    const Obsp* const obs_ptr = ptcls + pid;
    const Eigen::Vector2f this_obs(obs_ptr->x, obs_ptr->y);
    const float angle = obs_ptr->a;
    initialize(raw_segs, &this_obs, sid, flags);
    const int s_id = static_cast<int>(ceilf((ang_min + angle + M_PI) / ang_incre)) % full_rnum, 
        e_id = (s_id + range_num - 1) % full_rnum;
    const int th_num = blockDim.x;
    for (int i = 0; i < 16; i++) {       // 初始化深度（一个大值）
        const int cur_i = sid + i * th_num;
        if (cur_i >= range_num) {
            break;  // warp divergence 2
        }
        range[cur_i] = FLOAT_2048;
    }
    __syncthreads();
    if (flags[sid] == true) {           // warp divergence 1
        const float* const base = (raw_segs + 4 * sid);
        Eigen::Vector2f p1(*(base), *(base + 1)), p2(*(base + 2), *(base + 3));
        singleSegZbuffer(p1, p2, ptcls + pid, s_id, e_id, range_num, ang_incre, range);
    }
    __syncthreads();
    // 每个线程需要继续参与计算
    // 深度图计算完成之后，需要计算weight
    /// 每个block可以得到自己particle的深度图
    for (int i = 0; i < 16; i++) {       
        const int cur_i = sid + i * th_num;
        if (cur_i >= range_num) break;  // warp divergence 2
        if (single_flag) {
            weights[cur_i] = orderedIntToFloat(range[cur_i]);
        } else {
            const float val = orderedIntToFloat(range[cur_i]);
            const float abs_diff = abs(ref[cur_i] - val);
            float *pos = &weights[pid];
            atomicAdd_system(pos, abs_diff);
        }
    }
    __syncthreads();
}

__global__ void initTest(
    const Obsp* const ptcls,
    bool* flags
) {
    const int pid = blockIdx.x, sid = threadIdx.x;
    const Obsp* const obs_ptr = ptcls + pid;
    const Eigen::Vector2f this_obs(obs_ptr->x, obs_ptr->y);
    const float angle = obs_ptr->a;
    initialize(raw_segs, &this_obs, sid, flags);
    __syncthreads();
}
