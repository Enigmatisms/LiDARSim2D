#pragma once
#include <Eigen/Core>
#include <queue>

const double margin = 1e-6;

// 投影边
class Edge: public std::deque<Eigen::Vector3d> {
public:
    double min_dist = 1e9;
    bool valid = true;                      // 是否完全被遮挡
public:
    Edge() {reset();}

    bool angleInRange(double angle) const {
        if (this->front().z() > this->back().z())        // 跨越奇异
            return (angle > this->front().z() + margin) || (angle < this->back().z() - margin);
        else
            return (angle > this->front().z() + margin) && (angle < this->back().z() - margin);
    }
    
    bool angleInRange(double angle, int k, int k_1) const {
        const double k_ang = this->at(k).z(), k_ang_1 = this->at(k_1).z();
        if (k_ang > k_ang_1)        // 跨越奇异
            return (angle > k_ang + margin) || (angle < k_ang_1 - margin);
        else
            return (angle > k_ang + margin) && (angle < k_ang_1 - margin);
    }

    void reset() {
        clear();
        min_dist = 1e9;
        valid = true;
    }

    void initWithObs(Eigen::Vector2d obs);            // 根据观测点，初始化角度以及ids
    
    int rotatedBinarySearch(double angle) const;            // binary search acceleration

    Eigen::Vector2d getRayIntersect(const Eigen::Vector3d& ray, const Eigen::Vector2d& obs) const;
private:
    int binarySearch(std::pair<int, int> range, double angle) const;
};
