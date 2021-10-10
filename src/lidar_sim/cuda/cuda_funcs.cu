#include "cuda_funcs.h"

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0

__device__ __forceinline__ int floatToOrderedInt( float floatVal ) {
 int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}

__device__ void initialize(const float* const segments, const Obsp* const obs, int id, bool* flags) {
    const int base = 4 * id;
    const Eigen::Vector2d pt1(*(segments + base), *(segments + base + 1)), pt2(*(segments + base + 2), *(segments + base + 3));
    const Eigen::Vector2d norm(pt1.y() - pt2.y(), pt2.x() - pt1.x());
    const Eigen::Vector2d ctr_vec = (pt1 + pt2) / 2.0 - Eigen::Vector2d(obs->x, obs->y);
    if (ctr_vec.x() * norm.x() + ctr_vec.y() * norm.y() < 0.0)
        flags[id] = false;
    else flags[id] = true;
}

template<bool singl>
__device__ bool isIdInRange(const int& range_sid, const int& range_eid, const int& id) {
    if (singl == true)
        return (id >= range_sid) || (id <= range_eid);
    else
        return (id >= range_sid) && (id <= range_eid);
}

template<bool singl>
__device__ int getRangeOffset(const int& range_sid, const int& range_eid, const int& id, const int range_num) {
    if (singl == true)
        if (id <= range_eid)
            return range_num - range_eid + id;
    return id - range_sid;
}

__device__ float getRange(const Obsp& p1, const Obsp& vec_line, const Eigen::Vector2d& obs_pt, const double angle) {
    const Eigen::Vector2d vec(cos(angle), sin(angle));
    Eigen::Matrix2d A;
    A << vec_line.x, vec_line.y, -vec(0), -vec(1);
    const double b1 = -vec(1) * obs_pt.x() + vec(0) * obs_pt.y();
    const double b2 = -vec_line.y * p1.x + vec_line.x * p1.y;;
    const Eigen::Vector2d b(b1, b2);
    const double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    if (std::abs(det) < 1e-5) {
        return (Eigen::Vector2d(p1.x, p1.y) - obs_pt).norm();
    }
    return (A * b - obs_pt).norm();
}

__device__ void singleSegZbuffer(
    const Obsp& p1, const Obsp& p2, const Obsp* const ptcls,
    const int s_id, const int e_id, const int range_num,
    const double ang_incre, int* range
) {
    const Obsp obs(ptcls);
    const Obsp ray1 = p1 - obs, ray2 = p2 - obs, vec_line = p2 - p1;
    const double sang = atan2(ray1.y, ray1.x), eang = atan2(ray2.y, ray2.x);
    const int id_s = static_cast<int>(floor((sang + M_PI) / ang_incre)),
        id_e = static_cast<int>(floor((eang + M_PI) / ang_incre));
    const int max_ray_num = round(2 * M_PI / ang_incre);
    bool not_int_range = false;
    const Eigen::Vector2d obs_pt(obs.x, obs.y);
    const bool range_singl = (s_id > e_id), edge_singl = (id_s > id_e);
    if (range_singl) {          // 深度图范围角度奇异
        if (edge_singl == false) {      // 被投影边角度不奇异
            if (id_e < s_id && id_s > e_id)
                not_int_range = true;
        }                       // 奇异必定有重合部分
    } else {
        if (edge_singl) {      // 被投影边角度奇异
            if (id_e > e_id && id_s < s_id)
                not_int_range = true;
        } else {
            if (id_s > e_id || id_e < s_id)
                not_int_range = true;
        }
    }
    if (not_int_range == false) {   // 不进行动态并行，线程不够用，一个SM才2048个线程
        if (edge_singl) {
            for (int i = id_s; i < max_ray_num; i++) {
                if (range_singl)        // 超出range范围的不计算
                    if (isIdInRange<true>(s_id, e_id, i) == false) continue;
                else
                    if (isIdInRange<false>(s_id, e_id, i) == false) continue;
                const double angle = ang_incre * static_cast<double>(i) - M_PI;
                const float rval = getRange(p1, vec_line, obs_pt, angle);
                const int range_int  = floatToOrderedInt(rval);
                int offset = 0;
                if (range_singl) offset = getRangeOffset<true>(s_id, e_id, i, range_num);
                else offset = getRangeOffset<false>(s_id, e_id, i, range_num);
                int *pos = &range[offset];
                atomicMin(pos, range_int);         // 原子压入
            }
            for (int i = 0; i <= id_e; i++) {
                if (range_singl)        // 超出range范围的不计算
                    if (isIdInRange<true>(s_id, e_id, i) == false) continue;
                else
                    if (isIdInRange<false>(s_id, e_id, i) == false) continue;
                const double angle = ang_incre * static_cast<double>(i) - M_PI;
                const float rval = getRange(p1, vec_line, obs_pt, angle);
                int range_int  = floatToOrderedInt(rval);
                int offset = 0;
                if (range_singl) offset = getRangeOffset<true>(s_id, e_id, i, range_num);
                else offset = getRangeOffset<false>(s_id, e_id, i, range_num);
                int *pos = &range[offset];
                atomicMin(pos, range_int);         // 原子压入
            }
        } else {
            for (int i = id_s; i <= id_e; i++) {
                if (range_singl)        // 超出range范围的不计算
                    if (isIdInRange<true>(s_id, e_id, i) == false) continue;
                else
                    if (isIdInRange<false>(s_id, e_id, i) == false) continue;
                const double angle = ang_incre * static_cast<double>(i) - M_PI;
                const float rval = getRange(p1, vec_line, obs_pt, angle);
                const int range_int  = floatToOrderedInt(rval);
                int offset = 0;
                if (range_singl) offset = getRangeOffset<true>(s_id, e_id, i, range_num);
                else offset = getRangeOffset<false>(s_id, e_id, i, range_num);
                int *pos = &range[offset];
                atomicMin(pos, range_int);         // 原子压入
            }
        }
    }
}

/// 共享内存需要用在flags / range上
__global__ void particleFilter(
    const Obsp* const ptcls,
    const float* const raw_segs,
    const float* const ref, float* weights,
    const int s_id, const int e_id, const double ang_incre,
    const int range_num, const bool single_flag
) {
    extern __shared__ int range[];          //...一个数据类型分为了两个不同意义以及类型的块
    bool* flags = (bool*)(&range[range_num]);
    const int pid = blockIdx.x, sid = threadIdx.x;
    initialize(raw_segs, ptcls + pid, sid, flags);
    __syncthreads();
    const int th_num = blockDim.x;
    for (int i = 0; i < 4; i++) {       // 初始化深度（一个大值）
        const int cur_i = sid + i * th_num;
        if (cur_i >= range_num) break;  // warp divergence 2
        range[cur_i] = FLOAT_2048;
    }
    __syncthreads();
    if (flags[sid] == true) {           // warp divergence 1
        const float* const base = (raw_segs + 4 * sid);
        Obsp p1(*(base), *(base + 1)), p2(*(base + 2), *(base + 3));
        singleSegZbuffer(p1, p2, ptcls + pid, s_id, e_id, range_num, ang_incre, range);
    }
    __syncthreads();
    // 每个线程需要继续参与计算
    // 深度图计算完成之后，需要计算weight
    /// 每个block可以得到自己particle的深度图
    for (int i = 0; i < 4; i++) {       
        const int cur_i = sid + i * th_num;
        if (cur_i >= range_num) break;  // warp divergence 2
        if (single_flag) {
            weights[cur_i] = orderedIntToFloat(range[cur_i]);
        } else {
            float val = orderedIntToFloat(range[cur_i]);
            float abs_diff = abs(ref[cur_i] - val);
            float *pos = &weights[pid];
            atomicAdd(pos, abs_diff);
        }
    }
    __syncthreads();
    // 计算完每一个点的值
}
