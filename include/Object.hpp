#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Edge.hpp"
#include "LOG.hpp"

/// @ref https://stackoverflow.com/questions/8372918/c-std-list-sort-with-custom-comparator-that-depends-on-an-member-variable-for 
struct EdgeCompFunctor{
    EdgeCompFunctor(const std::vector<Edge>& _egs): egs(_egs) {
        LOG_GAY("Constructing functor: egs size: %lu, address: %x", _egs.size(), &egs);
    }
    const std::vector<Edge>& egs;
    bool operator() (const size_t& e1, const size_t& e2) const {
        return egs[e1].min_dist > egs[e2].min_dist;
    }
};

// 物体定义，物体内部有很多edge
class Object {
using HeapType = std::priority_queue<size_t, std::vector<size_t>, EdgeCompFunctor>;
public:
    Object(int _id = 0): id(_id){
        min_dist = 1e9;
    }
    ~Object() {}
public:
    void internalProjection(const Eigen::Vector2d& obs);

    void intialize(const std::vector<Eigen::Vector2d>& pts, const Eigen::Vector2d& obs);

    void visualizeEdges(cv::Mat& src, cv::Point obs) const;

    void makePolygons4Render(Eigen::Vector2d obs, std::vector<std::vector<cv::Point>>& polygons) const;

    void externalOcclusion(Object& obj, Eigen::Vector2d obs);
private:
    bool rangeSuitable(
        const Eigen::Vector3d& p1, 
        const Eigen::Vector3d& p2, 
        const Eigen::Vector2d& beam, 
        const Eigen::Vector2d& obs,
        Eigen::Vector3d& intersect
    ) const;

    void externalProjector(Edge& src, const Eigen::Vector2d& obs);

    /// @brief returns whether a false choice is being operated
    void projectEdge2Edge(const Edge& src, const Eigen::Vector2d& obs, Edge& dst, HeapType& heap);

    void breakEdge(Eigen::Vector2d b1, Eigen::Vector2d b2, Eigen::Vector2d obs, Edge& dst, HeapType& heap);

    static Eigen::Vector3d getIntersection(
        const Eigen::Vector2d& vec,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2, 
        const Eigen::Vector2d& obs
    );
public:
    int id;
    bool valid;
    double min_dist;
    std::vector<Edge> edges;
    std::pair<double, double> angle_range;
};