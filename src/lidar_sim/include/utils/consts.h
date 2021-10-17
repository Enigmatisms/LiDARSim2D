/// @author (Enigmatisms:https://github.com/Enigmatisms) @copyright Qianyue He
#pragma once
#include <array>
#include <Eigen/Core>
#include <opencv2/core.hpp>

typedef std::vector<Eigen::Vector3d> Wall;
typedef std::vector<Wall> Walls;

extern const std::array<cv::Point, 11> north_wall;      // 111
extern const std::array<cv::Point, 11> south_wall;
extern const std::array<cv::Point, 9> east_wall;       // 90
extern const std::array<cv::Point, 9> west_wall;

constexpr float RED[3] = {1.0, 0, 0}; 
constexpr float GREEN[3] = {0.0, 1.0, 0}; 
constexpr float BLUE[3] = {0.0, 0.0, 1.0}; 
constexpr float YELLOW[3] = {1.0, 1.0, 0.0}; 
constexpr float D_WHITE[3] = {0.4, 0.4, 0.4}; 
constexpr float L_GRAY[3] = {0.16, 0.16, 0.16}; 
constexpr float D_GRAY[3] = {0.04, 0.04, 0.04}; 
constexpr float BLACK[3] = {0.0, 0.0, 0.0}; 
constexpr float WHITE[3] = {1.0, 1.0, 1.0}; 

std::string getPackagePath();
