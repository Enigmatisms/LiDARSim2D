#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>

typedef std::vector<cv::Point> Obstacle;

void mapDraw(const std::vector<Obstacle>& obstalces, const Obstacle& obst, cv::Mat& src);

void mapLoad(std::string path, std::vector<std::vector<cv::Point>>& obstacles);

void mapSave(const std::vector<std::vector<cv::Point>>& obstacles, std::string path);
