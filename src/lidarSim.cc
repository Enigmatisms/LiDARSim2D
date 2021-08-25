#include "lidarSim.hpp"
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

LidarSim::LidarSim(Eigen::Vector3d angles): 
    sparse_min(angles.x()), sparse_max(angles.y()), angle_incre(angles.z()), sparse_incre(angle_incre * 5), rng(0)
{
    full_num = std::round(2 * M_PI / angle_incre);
    sparse_ray_num = std::round((sparse_max - sparse_min) / sparse_incre);       // 5点平均
    printf("Sparse ray num: %d\n", sparse_ray_num);
}

void LidarSim::scan(const std::vector<std::vector<cv::Point>>& obstacles, Eigen::Vector2d act_obs, cv::Mat& src, double angle) {
    cv::rectangle(src, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(src, floors, cv::Scalar(40, 40, 40), -1);
    cv::drawContours(src, obstacles, -1, cv::Scalar(10, 10, 10), -1);

    std::vector<double> act_range;
    act_range.resize(full_num, -1.0);
    Volume act_vol;
    std::vector<Edge> act_egs;
    act_vol.calculateVisualSpace(obstacles, cv::Point(act_obs.x(), act_obs.y()), src);
    act_vol.visualizeVisualSpace(obstacles, act_obs, src);
    act_vol.getValidEdges(act_egs);
    int angle_offset = static_cast<int>(floor(angle / angle_incre));
    for (const Edge& eg: act_egs)
        edgeIntersect(eg, act_obs, act_range);
    std::vector<double> sparse(sparse_ray_num);
    scanMakeSparse(act_range, sparse, angle_offset);
    visualizeRay(sparse, act_obs, src, angle);
}

void LidarSim::edgeIntersect(const Edge& eg, const Eigen::Vector2d& obs, std::vector<double>& range) const {
    double angle_start = eg.front().z(), angle_end = eg.back().z();
    int id_start = static_cast<int>(ceil((angle_start + M_PI) / angle_incre)), 
        id_end = static_cast<int>(floor((angle_end + M_PI) / angle_incre));
    if (id_start == id_end + 1) return;
    if (id_start > id_end) {            // 奇异角度
        for (int i = id_start; i < full_num; i++) {
            double angle = angle_incre * static_cast<double>(i) - M_PI;
            Eigen::Vector3d vec(cos(angle), sin(angle), angle);
            Eigen::Vector2d intersect = eg.getRayIntersect(vec, obs);
            range[i] = intersect.norm();
        }
        for (int i = 0; i <= id_end; i++) {
            double angle = angle_incre * static_cast<double>(i) - M_PI;
            Eigen::Vector3d vec(cos(angle), sin(angle), angle);
            Eigen::Vector2d intersect = eg.getRayIntersect(vec, obs);
            range[i] = intersect.norm();
        }
    } else {
        for (int i = id_start; i <= id_end; i++) {
            double angle = angle_incre * static_cast<double>(i) - M_PI;
            Eigen::Vector3d vec(cos(angle), sin(angle), angle);
            Eigen::Vector2d intersect = eg.getRayIntersect(vec, obs);
            range[i] = intersect.norm();
        }
    }
}

void LidarSim::visualizeRay(const std::vector<double>& range, const Eigen::Vector2d& obs, cv::Mat& dst, double now_ang) const {
    const cv::Point cv_obs(obs.x(), obs.y());
    for (int i = 0; i < sparse_ray_num; i++) {
        double angle = -M_PI + static_cast<double>(i) * sparse_incre + now_ang;
        Eigen::Vector2d ray = range[i] * Eigen::Vector2d(cos(angle), sin(angle)) + obs;
        cv::Point ray_end(ray.x(), ray.y());
        cv::line(dst, cv_obs, ray_end, cv::Scalar(0, 0, 255), 1);
    }
}

void LidarSim::scanMakeSparse(const std::vector<double>& range, std::vector<double>& sparse, int angle_offset){
    for (size_t i = 0; i < sparse.size(); i++) {
        double range_sum = 0.0;
        int i5 = 5 * i;
        for (size_t j = 0; j < 5; j++)
            range_sum += range[(i5 + j + angle_offset) % full_num];
        range_sum /= 5.0;
        range_sum += rng.gaussian(10.0 * range_sum / 400.0);
        sparse[i] = range_sum;
    }
}

