/// @author (Enigmatisms:https://github.com/Enigmatisms) @copyright Qianyue He
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>

typedef std::vector<cv::Point> Obstacle;

void mapDraw(const std::vector<Obstacle>& obstalces, const Obstacle& obst, cv::Mat& src);

void mapLoad(std::string path, std::vector<std::vector<cv::Point>>& obstacles);

void mapLoad(std::string path, std::vector<std::vector<Eigen::Vector3d>>& obstacles);

void mapSave(const std::vector<std::vector<cv::Point>>& obstacles, std::string path);

void extraTask(const std::array<cv::Point, 2>& task, cv::Mat& src, int mode_id);

void plotSpeedInfo(cv::Mat& src, double trans_vel, double act_vel);
