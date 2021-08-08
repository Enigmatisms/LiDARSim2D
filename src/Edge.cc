#include <Eigen/Dense>
#include "Edge.hpp"
#include "LOG.hpp"

const double bs_margin = 1e-7;

void Edge::initWithObs(Eigen::Vector2d obs, int second_id) {
    LOG_CHECK("Init with obs, this address is: %x, size is %lu", this, this->size());
    proj_ids.first = -1;                            // 打断式投影的第二个edge中少一个投影点
    if (second_id != -1)
        proj_ids.second = static_cast<int>(this->size()) - 1;
    else
        proj_ids.second = -1;
    Eigen::Vector2d front_beam = this->front().block<2, 1>(0, 0) - obs, back_beam = this->back().block<2, 1>(0, 0) - obs;
    int max_size = static_cast<int>(this->size()) - 1;
    min_dist = std::min(front_beam.norm(), back_beam.norm());
    for (int i = 1; i < max_size; i++) {
        Eigen::Vector2d vec = this->at(i).block<2, 1>(0, 0) - obs;
        min_dist = std::min(vec.norm(), min_dist);   
    }
    valid = true;
}

int Edge::rotatedBinarySearch(double angle) const{
    int s = 0, e = static_cast<int>(size()) - 1, m = static_cast<int>((s + e) / 2);
    double start_angle = front().z(), mid_angle = at(m).z();
    if (front().z() < back().z())
            return binarySearch(std::make_pair(0, e), angle);
    while (s < e) {
        if (mid_angle > start_angle) {
            s = m + 1;
            start_angle = mid_angle;
        } else if (mid_angle < start_angle){
            e = m;
        } else {
            if (at(e).z() < mid_angle)
                m = e;
            else m = s;
            break;
        }
        m = static_cast<int>((s + e) / 2);
        mid_angle = at(m).z();
    }
    if (angle >= front().z() - bs_margin) {
        int id = binarySearch(std::make_pair(0, m - 1), angle);
        if (id == -2) return m;
        return id;
    } 
    int id = binarySearch(std::make_pair(m, static_cast<int>(size()) - 1), angle);
    if (id == -1) return m;
    return id;
}
int Edge::binarySearch(std::pair<int, int> range, double angle) const{
    int s = range.first, e = range.second, m = static_cast<int>((s + e) / 2);
    if (angle < at(s).z() - bs_margin) return -1;
    else if (angle > at(e).z() + bs_margin) return -2;
    int range_min = std::max(1, range.first);
    double mid_angle = at(m).z();
    while (s < e) {
        if (mid_angle - bs_margin > angle) {
            e = m;
        } else if (mid_angle + bs_margin < angle) {
            s = m + 1;
        } else {
            return std::max(range_min, m);
        }
        m = static_cast<int>((s + e) / 2);
        mid_angle = at(m).z();
    } 
    return std::max(range_min, s);
}

Eigen::Vector3d getIntersection(
    const Eigen::Vector2d& vec,
    const Eigen::Vector3d& _p1,
    const Eigen::Vector3d& _p2, 
    const Eigen::Vector2d& obs
) {
    const Eigen::Vector2d p1 = _p1.block<2, 1>(0, 0), p2 = _p2.block<2, 1>(0, 0);
    const Eigen::Vector2d vec_line = p2 - p1;
    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    A << -vec(1), vec(0), -vec_line(1), vec_line(0);
    double b1 = Eigen::RowVector2d(-vec(1), vec(0)) * obs;
    double b2 = Eigen::RowVector2d(-vec_line(1), vec_line(0)) * p1;
    const Eigen::Vector2d b(b1, b2);
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    if (std::abs(det) < 1e-5)
        return _p1;
    const Eigen::Vector2d pt = A.inverse() * b, new_vec = pt - obs;
    double angle = atan2(new_vec(1), new_vec(0));
    Eigen::Vector3d result;
    result << pt, angle;
    return result;                   // 解交点
}

Eigen::Vector2d Edge::getRayIntersect(const Eigen::Vector3d& ray, const Eigen::Vector2d& obs) const {
    int id = rotatedBinarySearch(ray.z());
    if (id < 0) throw "Something went wrong with id < 0, which should not happen at all.\n";
    const Eigen::Vector2d p1 = at(id - 1).block<2, 1>(0, 0), p2 = at(id).block<2, 1>(0, 0);
    const Eigen::Vector2d vec_line = p2 - p1, vec = ray.block<2, 1>(0, 0);
    Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
    A << -vec(1), vec(0), -vec_line(1), vec_line(0);
    double b1 = Eigen::RowVector2d(-vec(1), vec(0)) * obs;
    double b2 = Eigen::RowVector2d(-vec_line(1), vec_line(0)) * p1;
    const Eigen::Vector2d b(b1, b2);
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    if (std::abs(det) < 1e-5)
        return p1;
    return A.inverse() * b - obs;
}
