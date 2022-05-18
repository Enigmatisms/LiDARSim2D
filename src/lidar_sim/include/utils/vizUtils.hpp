#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>

void drawScanLine(const Eigen::Vector2d& pos, const Eigen::Vector3d& lidar_param, const std::vector<float>& scan, double angle, cv::Mat& dst);

void drawObstacles(const std::vector<std::vector<cv::Point>>& _obstcs, const Eigen::Vector2d& obs, cv::Mat& dst);

void drawScannerPose(const Eigen::Vector2d& obs, double angle, cv::Mat& dst);

void cvPoints2Meshes(const std::vector<std::vector<cv::Point>>& pts, std::vector<std::vector<Eigen::Vector2d>>& meshes);