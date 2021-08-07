#include "mapEdit.h"

void mapDraw(const std::vector<Obstacle>& obstalces, const Obstacle& obst, cv::Mat& src) {
    for (size_t i = 0; i <  obstalces.size(); i++) {
        cv::drawContours(src, obstalces, i, cv::Scalar(80, 80, 80), -1);
        cv::drawContours(src, obstalces, i, cv::Scalar(255, 0, 0), 3);
        for (const cv::Point& pt: obstalces[i]) {
            cv::circle(src, pt, 2, cv::Scalar(0, 0, 0), -1);
        }
    }
    for (size_t i = 1; i < obst.size(); i++) {
        cv::line(src, obst[i - 1], obst[i], cv::Scalar(0, 255, 0), 3);
    }
    for (const cv::Point& pt: obst) {
        cv::circle(src, pt, 2, cv::Scalar(0, 0, 0), -1);
    }
}

void mapLoad(std::string path, std::vector<std::vector<cv::Point>>& obstacles) {
    std::ifstream file(path, std::ios::in);
    if (!file) {
        std::string error = "map of '" + path + "' does not exist.";
        throw error.c_str();
    }
    std::string s;
    while (getline(file, s)) {
        std::stringstream ss;
        ss << s;
        obstacles.emplace_back();
        Obstacle& obs = obstacles.back();
        size_t size = 0;
        ss >> size;
        for (size_t i = 0; i < size; i++) {
            cv::Point2d pt;
            ss >> pt.x >> pt.y;
            obs.push_back(pt);
        }
    }
}

void mapSave(const std::vector<std::vector<cv::Point>>& obstacles, std::string path) {
    std::ofstream file(path, std::ios::out);
    for (const Obstacle& obs: obstacles) {
        file << obs.size() << " ";
        for (const cv::Point& pt: obs) {
            file << pt.x << " " << pt.y << " ";
        }
        file << std::endl;
    }
    file.close();
}