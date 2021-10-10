#include <numeric>
#include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
#include "scanUtils.hpp"
#include "consts.h"
#include "cuda_pf.hpp"

const cv::Rect __walls(0, 0, 1200, 900);
const cv::Rect __floors(30, 30, 1140, 840);
constexpr int ALIGN_CHECK = 0x03;
constexpr double M_2PI = 2 * M_PI;

CudaPF::CudaPF(const cv::Mat& occ,  const Eigen::Vector3d& angles, int pnum): 
    occupancy(occ), point_num(pnum), angle_min(angles(0)), angle_max(angles(1)), angle_incre(angles(2)), rng(0)
{
    ray_num = static_cast<int>(floor((angles(1) - angles(0)) / angle_incre));
    full_ray_num = std::round(2 * M_PI / angle_incre);
    seg_num = 0;
    
    #ifdef CUDA_CALC_TIME
    for (int i = 0; i < 5; i++) {
        time_sum[i] = 0.0;
        cnt_sum[i] = 0.0;
    }
    #endif // CALC_TIME
}

void CudaPF::particleInitialize(const cv::Mat& src) {
    int pt_num = 0;
    particles.clear();
    while (pt_num < point_num) {
        const int x = rng.uniform(38, 1167);
        const int y = rng.uniform(38, 867);
        const double a = rng.uniform(0.0, M_2PI);
        if (src.at<uchar>(y, x) > 0x00) {
            particles.emplace_back(x, y, a);
            pt_num++;
        }
    }
    CUDA_CHECK_RETURN(cudaMemcpy(cu_pts, particles.data(), pt_num * sizeof(Obsp), cudaMemcpyHostToDevice));
}

void CudaPF::particleUpdate(double mx, double my, double angle) {
    for (Obsp& pt: particles) {
        double _mx = mx, _my = my, _a = angle;
        noisedMotion(_mx, _my, _a);
        pt.x += _mx;
        pt.y += _my;
        pt.a += _a;
        if (pt.a < 0)
            pt.a += M_2PI;
        else if (pt.a > M_2PI)
            pt.a -= M_2PI;
    }
    CUDA_CHECK_RETURN(cudaMemcpy(cu_pts, particles.data(), particles.size() * sizeof(Obsp), cudaMemcpyHostToDevice));
}

__host__ void CudaPF::intialize(const std::vector<std::vector<cv::Point>>& obstacles) {
    CUDA_CHECK_RETURN(cudaMalloc((void **) &weights, point_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &ref_range, ray_num * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &cu_pts, point_num * sizeof(Obsp)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &obs, sizeof(Obsp)));

    // 首先计算segment
    std::vector<std::vector<Eigen::Vector2d>> obstcs;
    for (const std::vector<cv::Point>& obstacle: obstacles) {            // 构建objects
        obstcs.emplace_back();
        for (const cv::Point& pt: obstacle)
            obstcs.back().emplace_back(pt.x, pt.y);
    }
    // ================ add walls ================
    obstcs.emplace_back();
    for (const cv::Point& pt: north_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 
    obstcs.emplace_back();
    for (const cv::Point& pt: east_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 
    obstcs.emplace_back();
    for (const cv::Point& pt: south_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 
    obstcs.emplace_back();
    for (const cv::Point& pt: west_wall)
        obstcs.back().emplace_back(pt.x, pt.y);
    seg_num = 0;
    std::vector<float> host_seg;
    for (const std::vector<Eigen::Vector2d>& obst: obstcs) {
        for (size_t i = 1; i < obst.size(); i++) {
            const Eigen::Vector2d& p = obst[i - 1];
            const Eigen::Vector2d& q = obst[i];
            host_seg.push_back(p.x());
            host_seg.push_back(p.y());
            host_seg.push_back(q.x());
            host_seg.push_back(q.y());
            seg_num ++;
        }
        const Eigen::Vector2d& p = obst.back();
        const Eigen::Vector2d& q = obst.front();
        host_seg.push_back(p.x());
        host_seg.push_back(p.y());
        host_seg.push_back(q.x());
        host_seg.push_back(q.y());
        seg_num ++;
    }
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(raw_segs, host_seg.data(), sizeof(float) * host_seg.size()));
    shared_to_allocate = sizeof(float) * ray_num + sizeof(bool) * seg_num;
    const int check_result = (shared_to_allocate & ALIGN_CHECK);
    if (check_result > 0)
        shared_to_allocate = shared_to_allocate + 4 - check_result;
    // 分配4的整数个字节 才能保证初始float数组的完整性
}


void CudaPF::filtering(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector3d act_obs, cv::Mat& src) {
    TicToc timer;
    cv::rectangle(src, __walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, __floors, cv::Scalar(40, 40, 40), -1);
    cv::drawContours(src, obstacles, -1, cv::Scalar(10, 10, 10), -1);
    // act_obs 的 z是角度，但是必须要0-2pi
    Obsp host_obs(act_obs(0), act_obs(1), (act_obs.z() < 0) ? act_obs.z() + M_2PI : act_obs.z());
    CUDA_CHECK_RETURN(cudaMemcpy(obs, &host_obs, sizeof(Obsp), cudaMemcpyHostToDevice));
    particleFilter <<< 1, seg_num, shared_to_allocate >>> (
                        obs, NULL, ref_range, angle_min, angle_incre, ray_num, full_ray_num, true);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    particleFilter <<< point_num, seg_num, shared_to_allocate >>> (
                        cu_pts, ref_range, weights, angle_min, angle_incre, ray_num, full_ray_num, false);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    std::vector<float> weight_vec(point_num);
    CUDA_CHECK_RETURN(cudaMemcpy(weight_vec.data(), weights, sizeof(float) * point_num, cudaMemcpyHostToDevice));
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < point_num; i++) {
        weight_vec[i] /= static_cast<float>(point_num);
        weight_vec[i] = 1.0 / (weight_vec[i] + 1.0);
    }
    
    float weight_sum = std::accumulate(weight_vec.begin(), weight_vec.end(), 0.0);
    for (float& val: weight_vec)                                  // 归一化形成概率
        val /= weight_sum;
    importanceResampler(weight_vec);                               // 重采样
    // visualizeParticles(weights, src);
}

/// @ref implementation from Thrun: Probabilistic Robotics
void CudaPF::importanceResampler(const std::vector<float>& weights) {
    std::vector<Obsp> tmp;
    std::vector<int> tmp_ids;
    double dpoint_num = static_cast<double>(point_num);
    double r = rng.uniform(0.0, 1.0 / dpoint_num);
    double c = weights.front();
    int i = 0;
    for (int m = 1; m <= point_num; m++) {
        double u = r + static_cast<double>(m - 1) / dpoint_num;
        while (u > c) {
            i++;
            c += weights[i];
        }
        tmp.push_back(particles[i]);
    }
    particles.assign(tmp.begin(), tmp.end());
}

void CudaPF::scanPerturb(std::vector<float>& range) {
    for (float& val: range)
        val += rng.gaussian(7);
}

void CudaPF::singleDebugDemo(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector3d act_obs, cv::Mat& src) {
    cv::rectangle(src, cv::Rect(0, 0, 1200, 900), cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, cv::Rect(30, 30, 1140, 840), cv::Scalar(40, 40, 40), -1);
    cv::drawContours(src, obstacles, -1, cv::Scalar(10, 10, 10), -1);
    int start_id = static_cast<int>(ceil((angle_min + act_obs(2) + M_PI) / angle_incre)) % full_ray_num, 
        end_id = (start_id + ray_num - 1) % full_ray_num;
    Obsp host_obs(act_obs(0), act_obs(1), (act_obs.z() < 0) ? act_obs.z() + M_2PI : act_obs.z());
    CUDA_CHECK_RETURN(cudaMemcpy(obs, &host_obs, sizeof(Obsp), cudaMemcpyHostToDevice));
    particleFilter <<< 1, seg_num, shared_to_allocate >>> (
                        obs, NULL, ref_range, angle_min, angle_incre, ray_num, full_ray_num, true);
    std::vector<float> weight_vec(point_num);
    std::vector<float> range(ray_num, 0.0);
    CUDA_CHECK_RETURN(cudaMemcpy(range.data(), ref_range, sizeof(float) * ray_num, cudaMemcpyHostToDevice));
    
    importanceResampler(weight_vec);                               // 重采样
    const cv::Point cv_obs(act_obs.x(), act_obs.y());
    for (int i = 0; i < ray_num; i++) {
        double angle = static_cast<double>(i + start_id) * angle_incre - M_PI;
        double rval = range[i];
        cv::Point2d trans(rval * cos(angle), rval * sin(angle));
        cv::Point lpt = cv_obs + cv::Point(trans.x, trans.y);
        cv::line(src, cv_obs, lpt, cv::Scalar(0, 0, 255), 1);
    }
}
