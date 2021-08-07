#pragma once
#include <Eigen/Core>
#include <queue>

const double margin = 1e-6;

// 投影边
class Edge: public std::deque<Eigen::Vector3d> {
public:
    std::pair<int, int> proj_ids;           // 待投影点的id
    double min_dist = 1e9;
    bool valid = true;                      // 是否完全被遮挡
public:
    Edge() {reset();}

    template<bool shrink = true>
    bool angleInRange(double angle) const {
        if (shrink) {
            if (this->front().z() > this->back().z()) {        // 跨越奇异
                return (angle > this->front().z() + margin) || (angle < this->back().z() - margin);
            } else {
                return (angle > this->front().z() + margin) && (angle < this->back().z() - margin);
            }
        } else {
            if (this->front().z() > this->back().z()) {        // 跨越奇异
                return (angle > this->front().z() - margin) || (angle < this->back().z() + margin);
            } else {
                return (angle > this->front().z() - margin) && (angle < this->back().z() + margin);
            }
        }
    }

    void reset() {
        clear();
        proj_ids.first = 0;
        proj_ids.second = 0;
        min_dist = 1e9;
        valid = true;
    }

    void initWithObs(Eigen::Vector2d obs, int second_id);            // 根据观测点，初始化角度以及ids
    
    int rotatedBinarySearch(double angle) const;            // binary search acceleration
private:
    int binarySearch(std::pair<int, int> range, double angle) const;
};
