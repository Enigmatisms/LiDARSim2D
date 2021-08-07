#pragma once
#include <fstream>
#include "Object.hpp"

typedef std::vector<cv::Point> Obstacle;

extern const cv::Rect walls;
extern const cv::Rect floors;

class Volume {
public:
    Volume(): heap(ObjCompFunctor(objs)) {}
    ~Volume() {}
public:
    void calculateVisualSpace(const std::vector<Obstacle>& _obstcs, cv::Point obs, cv::Mat& src);
    
    void visualizeVisualSpace(const std::vector<Obstacle>& _obstcs, const Eigen::Vector2d& obs, cv::Mat& dst) const;

    void simplePreVisualize(cv::Mat& src, const cv::Point& obs) const;

    void prettier(cv::Mat& src, const Eigen::Vector2d& obs) const;

    void reset() {
        while (heap.empty() == false) heap.pop();
        objs.clear();
    }
private:
    struct ObjCompFunctor{
        ObjCompFunctor(const std::vector<Object>& _objs): objs(_objs) {}
        const std::vector<Object>& objs;
        bool operator() (size_t o1, size_t o2) const {
            return objs[o1].min_dist > objs[o2].min_dist;
        }
    };

    std::priority_queue<size_t, std::vector<size_t>, ObjCompFunctor> heap;
    std::vector<Object> objs;
}; 