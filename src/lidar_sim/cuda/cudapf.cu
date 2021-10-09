#include "cudapf.h"

constexpr int FLOAT_2048 = 0xc5000000;      // when convert to float directly, this is equal to 2048.0

__device__ __forceinline__ int floatToOrderedInt( float floatVal ) {
 int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal ^ 0x80000000 : intVal ^ 0xFFFFFFFF;
}

__device__ __forceinline__ float orderedIntToFloat( int intVal ) {
    return __int_as_float( (intVal >= 0) ? intVal ^ 0xFFFFFFFF : intVal ^ 0x80000000);
}

/**
 * @brief GPU初始化，负责背面剔除（生成一个flag数组，block共享）
 * @note 此前需要使用vector取出所有片段，才能并行执行，片段直接保存在GPU常量内存中
 * @note 本函数每个线程计算一个是否valid的信息（根据法向量），保存到float数组
 * @param segments constant memory片段信息
 * @param flags 共享内存flags, x, y是观测点位置
 */
__device__ void initialize(const float* const segments, const Obsp* const obs, int id, bool* flags) {

}

/**
 * @brief 并行的z buffer, 渲染深度图
 * @note 输入线段的两个端点，进行“片段着色”
 * @param range 是共享内存的range，只有Z buffer全部计算完（并且转换为int） 才可以开始覆盖
 */
__device__ void singleSegZbuffer(
    const Obsp& p1, const Obsp& p2, const Obsp* const ptcls,
    const int s_id, const int e_id, const double ang_incre, int* range
) {
    Obsp ray1 = p1 - *(ptcls), ray2 = p2 - *(ptcls);
    const double sang = atan2(ray1.y, ray1.x), eang = atan2(ray2.y, ray2.x);
    const int id_s = static_cast<int>(floor((sang + M_PI) / ang_incre)),
        id_e = static_cast<int>(floor((eang + M_PI) / ang_incre));
    bool not_int_range = false;
    if (s_id > e_id) {          // 深度图范围角度奇异
        if (id_s < id_e) {      // 被投影边角度不奇异
            if (id_e < s_id && id_s > e_id)
                not_int_range = true;
        }                       // 奇异必定有重合部分
    } else {
        if (id_s > id_e) {      // 被投影边角度奇异
            if (id_e > e_id && id_s < s_id)
                not_int_range = true;
        } else {
            if (id_s > e_id || id_e < s_id)
                not_int_range = true;
        }
    }
    if (not_int_range == false) {
        // 不进行动态并行了，线程不够用，一个SM才2048个线程，那平均一个block 320个线程，也只有18block并行
        // 直接求解即可，仿照edge
    }
}

/// 共享内存需要用在flags / range上
__global__ void particleFilter(
    const Obsp* const ptcls,
    const float* const raw_segs,
    const float* const ref, float* weights,
    const int s_id, const int e_id, const double ang_incre,
    int range_num
) {
    extern __shared__ int range[];          //...一个数据类型分为了三个不同意义以及类型的块
    float* ref_ptr = (float*)(&range[range_num]);
    bool* flags = (bool*)(&range[range_num << 1]);
    const int pid = blockIdx.x, sid = threadIdx.x;
    initialize(raw_segs, ptcls + pid, sid, flags);
    __syncthreads();
    if (flags[sid] == true) {           // warp divergence 1
        const float* const base = (raw_segs + 4 * sid);
        Obsp p1(*(base), *(base + 1)), p2(*(base + 2), *(base + 3));
        singleSegZbuffer(p1, p2, ptcls + pid, s_id, e_id, ang_incre, range);
    }
    __syncthreads();
    // 每个线程需要继续参与计算
    // 深度图计算完成之后，需要计算weight
    /// 每个block可以得到自己particle的深度图
    const int th_num = blockDim.x;
    for (int i = 0; i < 4; i++) {       
        const int cur_i = sid + i * th_num;
        if (cur_i >= range_num) break;  // warp divergence 2
        float val = orderedIntToFloat(range[cur_i]);
        float abs_diff = abs(ref[cur_i] - val);
        atomicAdd(weights[pid], abs_diff);
    }
    __syncthreads();
    // 计算完每一个点的值
}