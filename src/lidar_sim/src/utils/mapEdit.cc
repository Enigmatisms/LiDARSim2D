#include <array>
#include "utils/mapEdit.h"

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

void extraTask(const std::array<cv::Point, 2>& task, cv::Mat& src, int mode_id) {
    if (task[0].x < 0 || task[1].x < 0) return;
    if (mode_id == 0) {
        cv::line(src, task[0], task[1], cv::Scalar(0, 255, 255), 3);
    } else if (mode_id == 1) {
        cv::rectangle(src, task[0], task[1], cv::Scalar(0, 255, 255), 3);
    } else {
        const cv::Point diff = task[0] - task[1];
        int radius = std::sqrt(diff.dot(diff));
        cv::circle(src, task[0], radius, cv::Scalar(0, 255, 255), 3);
    }
}

void mapLoad(std::string path, std::vector<std::vector<cv::Point>>& obstacles) {
    std::ifstream file(path, std::ios::in);
    if (!file) {
        system("pwd");
        std::string error = "map of '" + path + "' does not exist.";
        std::cerr << error << std::endl;
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

void plotSpeedInfo(cv::Mat& src, double trans_vel, double act_vel) {
    cv::rectangle(src, cv::Rect(700, 5, 104, 20), cv::Scalar(50, 50, 50), -1);
    int length = int(100 * (trans_vel - 1.0));
    cv::rectangle(src, cv::Rect(702, 7, length, 16), cv::Scalar(0, 255, 0), -1);
    std::string str = "set speed";
    cv::putText(src, str.c_str(), cv::Point(810, 15),
					cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(150, 150, 150));
    cv::rectangle(src, cv::Rect(960, 5, 104, 20), cv::Scalar(50, 50, 50), -1);
    length = int(50 * (act_vel - 0.0));
    cv::rectangle(src, cv::Rect(962, 7, length, 16), cv::Scalar(0, 255, 255), -1);
    str = "act speed";
    cv::putText(src, str.c_str(), cv::Point(1070, 15),
					cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(150, 150, 150));
}
