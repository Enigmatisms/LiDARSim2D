#include <numeric>
#include "cuda/host_func.hpp"
#include "utils/scanUtils.hpp"

short *sid_ptr = nullptr, *eid_ptr = nullptr;
float *all_segments = nullptr, *angles_ptr = nullptr, *dists_ptr = nullptr, *final_dense_ranges, *final_sparse_ranges, *oct_ranges;
bool *flag_ptr = nullptr;
size_t total_seg_num = 0;

__host__ void intializeFixed(int num_ray) {
    CUDA_CHECK_RETURN(cudaMallocHost((void **) &all_segments, 8192 * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &final_dense_ranges, num_ray * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &final_sparse_ranges, num_ray / 3 * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &oct_ranges, (num_ray << 3) * sizeof(float)));
}

__host__ void deallocateFixed() {
    CUDA_CHECK_RETURN(cudaFreeHost(all_segments));
    CUDA_CHECK_RETURN(cudaFree(final_dense_ranges));
    CUDA_CHECK_RETURN(cudaFree(final_sparse_ranges));
    CUDA_CHECK_RETURN(cudaFree(oct_ranges));
}

__host__ void deallocateDevice() {
    CUDA_CHECK_RETURN(cudaFree(sid_ptr));
    CUDA_CHECK_RETURN(cudaFree(eid_ptr));
    CUDA_CHECK_RETURN(cudaFree(angles_ptr));
    CUDA_CHECK_RETURN(cudaFree(dists_ptr));
    CUDA_CHECK_RETURN(cudaFree(flag_ptr));
}

__host__ void unwrapMeshes(const Meshes& meshes, bool initialized) {
    size_t mesh_point_cnt = 0;
    total_seg_num = 0;
    for (const Mesh& m: meshes) {
        size_t max_size = m.size() - 1;
        all_segments[mesh_point_cnt++] = m.front().x();
        all_segments[mesh_point_cnt++] = m.front().y();
        for (size_t i = 1; i < max_size; i++) {
            const Eigen::Vector2d& p = m[i];
            float x = p.x(), y = p.y();
            all_segments[mesh_point_cnt++] = x;
            all_segments[mesh_point_cnt++] = y;
            all_segments[mesh_point_cnt++] = x;
            all_segments[mesh_point_cnt++] = y;
        }
        all_segments[mesh_point_cnt++] = m.back().x();
        all_segments[mesh_point_cnt++] = m.back().y();
        total_seg_num += (m.size() - 1);
    }
    updateSegments(all_segments, mesh_point_cnt << 2);
    if (initialized == true)
        deallocateDevice();

    CUDA_CHECK_RETURN(cudaMalloc((void **) &sid_ptr, total_seg_num * sizeof(short)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &eid_ptr, total_seg_num * sizeof(short)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &angles_ptr, total_seg_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &dists_ptr, total_seg_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &flag_ptr, total_seg_num * sizeof(bool)));
}

// 外部判定，如果激光线数不是120的整数倍，则报错（所求的深度图将是线数 * 3，由于需要模拟深度不连续）
__host__ double rayTraceRenderCpp(const Eigen::Vector3d& lidar_param, const Eigen::Vector2d& pose, float angle, int ray_num, std::vector<float>& range) {
    // 对于静态地图而言，由于场景无需频繁update，unwrapMeshes函数调用频率低，则可以省略内存allocation操作
    const int lidar_ray_blocks = ray_num / DEPTH_DIV_NUM;
    const short num_blocks = static_cast<short>(ceilf(total_seg_num / 256.f));          // 面片数 / 128
    TicToc timer;
    timer.tic();
    preProcess<<<num_blocks, 256>>>(sid_ptr, eid_ptr, angles_ptr, dists_ptr, flag_ptr, ray_num, 
                total_seg_num, lidar_param.x(), lidar_param.z(), pose.x(), pose.y(), angle);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    double result = timer.toc();
    cudaStream_t streams[8];
    for (short i = 0; i < 8; i++)
        cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
    const short segment_per_block = static_cast<short>(ceil(0.125f * total_seg_num)),
                last_block_seg_num = short(total_seg_num) - 7 * segment_per_block;
    const size_t size_t_sblock = static_cast<size_t>(segment_per_block);
    const size_t shared_mem_size = (size_t_sblock * 13) + (DEPTH_DIV_NUM << 2) + 4 - size_t_sblock % 4;     // 13 = 8 + 4 + 1 = (4B angles 4B dists, 2B * 2 ids, 1B flags)
    for (int i = 0; i < 8; i++) {
        // 最后由于bool是单字节的类型，需要padding到4的整数倍字节数
        // local segments大小应该是 4B * (angles + dists) / 8 + DEPTH_DIV_NUM * 4B (深度图分区) + 2B * (sids + eids) / 8 + (1B * len(flags) / 8) + padding
        rayTraceKernel<<<lidar_ray_blocks, DEPTH_DIV_NUM, shared_mem_size, streams[i]>>>(
            sid_ptr, eid_ptr, angles_ptr, dists_ptr, flag_ptr, i, segment_per_block, 
            ((i < 7) ? segment_per_block : last_block_seg_num), &oct_ranges[i * ray_num], lidar_param.x(), lidar_param.z(), angle
        );
    }
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++)
        cudaStreamDestroy(streams[i]);
    getMininumRangeKernel<<<lidar_ray_blocks, DEPTH_DIV_NUM>>>(oct_ranges, final_dense_ranges, ray_num);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    sparsifyScan<<<lidar_ray_blocks, DEPTH_DIV_NUM / 3>>>(final_dense_ranges, final_sparse_ranges);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // 注意range的大小应该提前确定
    CUDA_CHECK_RETURN(cudaMemcpy(range.data(), final_sparse_ranges, sizeof(float) * ray_num / 3, cudaMemcpyDeviceToHost));
    return result;
}
