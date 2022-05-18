#include "utils/vizUtils.hpp"
#include "utils/consts.h"

void drawScanLine(
    const Eigen::Vector2d& pos, const Eigen::Vector3d& lidar_param, const std::vector<float>& scan, double angle, cv::Mat& dst
) {
    double angle_base = angle + lidar_param.x();
    for (size_t i = 0; i < scan.size(); ++i) {
        float r = scan[i];
        if (r > 1e5) continue;
        double angle = lidar_param.z() * double(i) + angle_base;
        Eigen::Vector2d dir(cos(angle), sin(angle));
        Eigen::Vector2d end_pos = pos + dir * r;
        cv::line(dst, cv::Point(pos.x(), pos.y()), cv::Point(end_pos.x(), end_pos.y()), cv::Scalar(0, 0, 255), 1);
    }
}

void drawObstacles(const std::vector<std::vector<cv::Point>>& _obstcs, const Eigen::Vector2d& obs, cv::Mat& dst) {
    cv::rectangle(dst, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(dst, floors, cv::Scalar(40, 40, 40), -1);
    cv::drawContours(dst, _obstcs, -1, cv::Scalar(10, 10, 10), -1);
}

void drawScannerPose(const Eigen::Vector2d& obs, double angle, cv::Mat& dst) {
    cv::circle(dst, cv::Point(obs.x(), obs.y()), 4, cv::Scalar(0, 255, 255), -1);
    cv::circle(dst, cv::Point(obs.x(), obs.y()), 7, cv::Scalar(40, 40, 40), 2);
    cv::Point start(obs.x(), obs.y()), end(obs.x() + 20 * cos(angle), obs.y() + 20 * sin(angle));
    cv::arrowedLine(dst, start, end, cv::Scalar(255, 0, 0), 2);
}

void cvPoints2Meshes(const std::vector<std::vector<cv::Point>>& pts, std::vector<std::vector<Eigen::Vector2d>>& meshes) {
    meshes.reserve(pts.size());
    for (const std::vector<cv::Point>& chain: pts) {
        meshes.emplace_back();
        std::vector<Eigen::Vector2d>& back_vec = meshes.back();
        back_vec.reserve(chain.size() + 1);
        for (const cv::Point& pt: chain)
            back_vec.emplace_back(pt.x, pt.y);
        back_vec.emplace_back(chain.front().x, chain.front().y);
    }

    meshes.emplace_back();
    meshes.back().reserve(north_wall.size() + 1);
    for (const cv::Point& pt: north_wall)
        meshes.back().emplace_back(pt.x, pt.y);
    meshes.emplace_back();
    meshes.back().reserve(east_wall.size() + 1);
    for (const cv::Point& pt: east_wall)
        meshes.back().emplace_back(pt.x, pt.y);
    meshes.emplace_back();
    meshes.back().reserve(south_wall.size() + 1);
    for (const cv::Point& pt: south_wall)
        meshes.back().emplace_back(pt.x, pt.y);
    meshes.emplace_back();
    meshes.back().reserve(west_wall.size() + 1);
    for (const cv::Point& pt: west_wall)
        meshes.back().emplace_back(pt.x, pt.y);
}
