#include "cuda/cuda_funcs.h"
#include "cuda/cuda_err_check.hpp"
#include <sm_60_atomic_functions.h>

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0
constexpr float _M_PI = M_PI;
constexpr float _M_2PI = 2 * _M_PI;
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

__device__ __forceinline__ void initialize(const float* const segments, const Vec2f* const obs, short id, bool* flags) {
    const float* ptr = segments + (id << 2);
    const Vec2f pt1(ptr[0], ptr[1]), pt2(ptr[2], ptr[3]);
    const Vec2f norm(pt1.y - pt2.y, pt2.x - pt1.x);
    const Vec2f ctr_vec = (pt1 + pt2) * 0.5 - Vec2f(obs->x, obs->y);
    if (ctr_vec.x * norm.x + ctr_vec.y * norm.y > 0.0) {
        flags[id] = false;
    } else {
        flags[id] = true;
    }
}

template<bool singl>
__device__ __forceinline__ bool isIdInRange(const short range_sid, const short range_eid, const short id) {
    if (singl == true)
        return (id >= range_sid) || (id <= range_eid);
    else
        return (id >= range_sid) && (id <= range_eid);
}

__device__ __forceinline__ short getRangeOffset(const short range_sid, const short id, const short range_num) {
    return min(id - range_sid, range_num);
}

__device__ __forceinline__ short getRangeOffsetSingl(const short range_sid, const short range_eid, const short id, const short range_num) {
    if (id <= range_eid)
        return max(range_num - range_eid + id, 0);
    return min(id - range_sid, range_num);
}

__device__ __forceinline__ float getRange(const Vec2f& p1, const Vec2f& vec_line, const Vec2f& obs_pt, const float angle, const float b2) {
    const float A00 = vec_line.x, A01 = -cosf(angle), A10 = vec_line.y, A11 = -sinf(angle);
    const float b1 = A11 * obs_pt.x - A01 * obs_pt.y;
    const float det = A00 * A11 - A01 * A10;
    if (abs(det) < 1e-5f) {
        return (p1 - obs_pt).norm();
    }
    const float det_1 = 1.0f / det;
    const Vec2f tmp(A00 * b1 + A01 * b2, A10 * b1 + A11 * b2);
    return (tmp * det_1 - obs_pt).norm();
}

__device__ void singleSegZbuffer(
    const Vec2f& p1, const Vec2f& p2, const Obsp* const ptcls,
    const short s_id, const short e_id, const int range_num,
    const float ang_incre, int* range
) {
    const Vec2f obs(ptcls->x, ptcls->y);
    const Vec2f ray1 = p1 - obs, ray2 = p2 - obs, vec_line = p2 - p1;
    const float sang = atan2f(ray1.y, ray1.x), eang = atan2f(ray2.y, ray2.x), b2 = -vec_line.y * p1.x + vec_line.x * p1.y;
    const short id_s = static_cast<short>(ceilf((sang + _M_PI) / ang_incre)),
        id_e = static_cast<short>(floorf((eang + _M_PI) / ang_incre));
    if (id_s == id_e + 1) {
        return;
    }
    const short max_ray_num = roundf(_M_2PI / ang_incre);
    const short range_num_1 = range_num - 1;
    const bool range_singl = (s_id > e_id), edge_singl = (id_s > id_e);
    if (range_singl) {          // 深度图范围角度奇异
        if (edge_singl == false) {      // 被投影边角度不奇异
            if (id_e < s_id && id_s > e_id) {
                return;
            }
        }                       // 奇异必定有重合部分
    } else {
        if (edge_singl) {      // 被投影边角度奇异
            if (id_e > e_id && id_s < s_id) {
                return;
            }
        } else {
            if (id_s > e_id || id_e < s_id) {
                return;
            }
        }
    }
    float angle = ang_incre * static_cast<float>(id_s) - _M_PI;
    if (edge_singl) {
        for (short i = id_s; i < max_ray_num; i++) {
            if (range_singl) {       // 超出range范围的不计算
                if (isIdInRange<true>(s_id, e_id, i) == false) {
                    angle += ang_incre;
                    continue;
                }
            } else {
                if (isIdInRange<false>(s_id, e_id, i) == false) {
                    angle += ang_incre;
                    continue;
                }
            }
            const float rval = getRange(p1, vec_line, obs, angle, b2);
            const int range_int = floatToOrderedInt(rval);
            short offset = 0;
            if (range_singl) {
                offset = getRangeOffsetSingl(s_id, e_id, i, range_num_1);
            } else {
                offset = getRangeOffset(s_id, i, range_num_1);
            }
            atomicMin(range + offset, range_int);         // 原子压入
            angle += ang_incre;
        }
        angle = _M_PI;
        for (short i = 0; i <= id_e; i++) {
            if (range_singl) {       // 超出range范围的不计算
                if (isIdInRange<true>(s_id, e_id, i) == false) {
                    angle += ang_incre;
                    continue;
                }
            } else {
                if (isIdInRange<false>(s_id, e_id, i) == false) {
                    angle += ang_incre;
                    continue;
                }
            }
            const float rval = getRange(p1, vec_line, obs, angle, b2);
            const int range_int = floatToOrderedInt(rval);
            short offset = 0;
            if (range_singl) {
                offset = getRangeOffsetSingl(s_id, e_id, i, range_num_1);
            } else {
                offset = getRangeOffset(s_id, i, range_num_1);
            }
            atomicMin(range + offset, range_int);         // 原子压入
            angle += ang_incre;
        }
    } else {
        for (short i = id_s; i <= id_e; i++) {
            if (range_singl) {       // 超出range范围的不计算
                if (isIdInRange<true>(s_id, e_id, i) == false) {
                    angle += ang_incre;
                    continue;
                }
            } else {
                if (isIdInRange<false>(s_id, e_id, i) == false) {
                    angle += ang_incre;
                    continue;
                }
            }
            const float rval = getRange(p1, vec_line, obs, angle, b2);
            const int range_int = floatToOrderedInt(rval);
            short offset = 0;
            if (range_singl) {
                offset = getRangeOffsetSingl(s_id, e_id, i, range_num_1);
            } else {
                offset = getRangeOffset(s_id, i, range_num_1);
            }
            atomicMin(range + offset, range_int);         // 原子压入
            angle += ang_incre;
        }
    }
}

/// 共享内存需要用在flags / range上
__global__ void particleFilter(
    const Obsp* const ptcls,
    // const float* const raw_segs,
    const float* const ref, float* weights,
    const float ang_min, const float ang_incre, const int range_num, 
    const int full_rnum, const int offset, const bool single_flag
) {
    extern __shared__ int range[];          //...一个数据类型分为了两个不同意义以及类型的块
    bool* flags = (bool*)(&range[range_num]);
    const int pid = blockIdx.x + offset, sid = threadIdx.x;
    const Obsp* const obs_ptr = ptcls + pid;
    const Vec2f this_obs(obs_ptr->x, obs_ptr->y);
    const float angle = obs_ptr->a;
    initialize(raw_segs, &this_obs, sid, flags);
    const int s_id = static_cast<int>(ceilf((ang_min + angle + _M_PI) / ang_incre)) % full_rnum, 
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
        const float* const base = (raw_segs + (sid << 2));
        Vec2f p1(*(base), *(base + 1)), p2(*(base + 2), *(base + 3));
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
