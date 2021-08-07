#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include "Object.hpp"

const double PI = 3.141592653589793238;
 
void Object::internalProjection(const Eigen::Vector2d& obs) {
    HeapType heap(edges);
    for (size_t i = 0; i < edges.size(); i++) {
        if (edges[i].valid)
            heap.emplace(i);
    }
    while (heap.empty() == false) {
        size_t top = heap.top();
        Edge& this_edge = edges[top];
        heap.pop();
        for (size_t i = 0; i < edges.size(); i++) {
            Edge& eg = edges[i];
            if (eg.valid == false || &eg == &this_edge)
                continue;
            projectEdge2Edge(this_edge, obs, eg, heap);
        }
    }
}

void Object::externalOcclusion(Object& obj, Eigen::Vector2d obs) {
    HeapType heap(edges);
    for (size_t i = 0; i < edges.size(); i++) {
        if (edges[i].valid)
            heap.emplace(i);
    }
    while (heap.empty() == false) {
        size_t top = heap.top();
        Edge& this_edge = edges[top];
        heap.pop();
        obj.externalProjector(this_edge, obs);
    }
}

void Object::externalProjector(Edge& src, const Eigen::Vector2d& obs) {
    HeapType heap(edges);
    for (size_t i = 0; i < edges.size(); i++)
        heap.emplace(i);
    while (heap.empty() == false) {
        size_t top = heap.top();
        Edge& this_edge = edges[top];
        heap.pop();
        if (this_edge.valid == false) continue;
        int fx = this_edge.front().x(), fy = this_edge.front().y(), bx = this_edge.back().x(), by = this_edge.back().y();
        projectEdge2Edge(src, obs, this_edge, heap);
    }
    bool valid_flag = false;
    for (size_t i = 0; i < edges.size(); i++) {
        if (edges[i].valid == true) {
            valid_flag = true;
            break;
        }
    }
    if (valid_flag == false)                    // 假如全部被遮挡，那么这个object就没有投影的必要了
        valid = false;
}

void Object::intialize(const std::vector<Eigen::Vector2d>& pts, const Eigen::Vector2d& obs) {
    edges.clear();
    Edge to_add;
    bool zero_pushed = false;
    for (size_t i = 1; i < pts.size(); i++) {
        Eigen::Vector2d vec = pts[i] - pts[i - 1];
        Eigen::Vector2d ctr_vec = (pts[i] + pts[i - 1]) / 2.0 - obs;
        Eigen::Vector2d norm = Eigen::Vector2d(-vec(1), vec(0));
        if (ctr_vec.dot(norm) < 0.0) {
            if (i == 1)
                zero_pushed = true;
            Eigen::Vector2d o2p = pts[i - 1] - obs;
            to_add.emplace_back();
            to_add.back() << pts[i - 1], atan2(o2p(1), o2p(0));
        } else {
            if (to_add.empty() == false) {
                Eigen::Vector2d o2p = pts[i - 1] - obs;
                to_add.emplace_back();
                to_add.back() << pts[i - 1], atan2(o2p(1), o2p(0));
                edges.push_back(to_add);
                to_add.reset();
            }
        }
    }
    Eigen::Vector2d vec = pts.front() - pts.back();
    Eigen::Vector2d ctr_vec = (pts.back() + pts.front()) / 2.0 - obs;
    Eigen::Vector2d norm = Eigen::Vector2d(-vec(1), vec(0));
    if (to_add.empty() == false) {
        Eigen::Vector2d o2p = pts.back() - obs;
        to_add.emplace_back();
        to_add.back() << pts.back(), atan2(o2p(1), o2p(0));
    }
    if (ctr_vec.dot(norm) < 0.0) {
        if (zero_pushed == true) {
            Edge& front = edges.front();
            if (to_add.empty() == true) {
                front.emplace_front();
                Eigen::Vector2d o2p = pts.back() - obs;
                front.front() << pts.back(), atan2(o2p(1), o2p(0));
            } else {
                for (Edge::const_reverse_iterator rit = to_add.crbegin(); rit != to_add.crend(); rit++)
                    front.push_front(*rit);
            }
        } else {
            if (to_add.size() == 0) {
                Eigen::Vector2d o2p = pts.back() - obs;
                to_add.emplace_back();
                to_add.back() << pts.back(), atan2(o2p(1), o2p(0));  
            }
            Eigen::Vector2d o2p = pts.front() - obs;
            to_add.emplace_back();
            to_add.back() << pts.front(), atan2(o2p(1), o2p(0));
            edges.push_back(to_add);
        }
    } else if (to_add.empty() == false) {
        edges.push_back(to_add);
    }
    for (Edge& eg: edges) {
        for (const Eigen::Vector3d& pt: eg) {
            double dist = (pt.block<2, 1>(0, 0) - obs).norm();
            if (dist < eg.min_dist) eg.min_dist = dist;
        }
    }
    for (size_t i = 0; i < edges.size(); i++) {
        Edge& eg = edges[i];
        eg.proj_ids.first = 0;
        eg.proj_ids.second = static_cast<int>(eg.size()) - 1;
        eg.valid = true;
        if (eg.min_dist < min_dist)
            min_dist = eg.min_dist;
    }
    valid = true;
}

void Object::makePolygons4Render(Eigen::Vector2d obs, std::vector<std::vector<cv::Point>>& polygons) const{
    for (const Edge& eg: edges) {
        if (eg.valid == false) continue;
        polygons.emplace_back();
        std::vector<cv::Point>& back = polygons.back();
        back.emplace_back(obs.x(), obs.y());
        for (const Eigen::Vector3d& p: eg) {
            back.emplace_back(p.x(), p.y());
        }
    }
}

void Object::projectEdge2Edge(const Edge& src, const Eigen::Vector2d& obs, Edge& dst, HeapType& heap) {
    int first_id = src.proj_ids.first, second_id = src.proj_ids.second, point_num = 0;
    std::array<bool, 2> can_proj = {false, false};
    if (first_id >= 0)  {
        can_proj[0] = true;
        point_num++;
    }    
    if (second_id >= 0) {
        can_proj[1] = true;
        point_num++;
    }
    if (first_id < 0 && second_id < 0) {
        first_id = 0;
        second_id = static_cast<int>(src.size()) - 1;
        point_num = 2;
    }
    bool pop_back_flag = true;
    Eigen::Vector2d beam = Eigen::Vector2d::Zero();
    double angle = 0.0;
    if (point_num == 2) {
        Eigen::Vector2d fpt = src[first_id].block<2, 1>(0, 0) - obs, ept = src[second_id].block<2, 1>(0, 0) - obs;
        double f_ang = src[first_id].z(), e_ang = src[second_id].z();
        bool head_in_range = dst.angleInRange(f_ang), end_in_range = dst.angleInRange(e_ang);
        if (head_in_range && end_in_range) {        // 需要有加edge逻辑
            LOG_ERROR_STREAM("Breaking edge!");
            breakEdge(fpt, ept, obs, dst, heap);
            return;
        } else if (head_in_range) {
            beam = fpt;
            angle = f_ang;
        } else if (end_in_range) {
            beam = ept;
            angle = e_ang;
            pop_back_flag = false;          // pop_front
        } else {                            // SRC的两个端点不在DST范围内（有可能完全遮挡的）
            const Eigen::Vector2d dst_f = dst.front().block<2, 1>(0, 0) - obs, dst_b = dst.back().block<2, 1>(0, 0) - obs;
            if (dst_f.dot(fpt) < 0.0 && dst_b.dot(ept) < 0.0) return;
            int f_id = src.rotatedBinarySearch(dst.front().z());
            if (f_id > 0) {
                Eigen::Vector3d tmp;
                bool dst_closer = rangeSuitable(src[f_id - 1], src[f_id], dst_f, obs, tmp);
                if (dst_closer == true) return;
            }
            int e_id = src.rotatedBinarySearch(dst.back().z());
            if (e_id > 0) {
                Eigen::Vector3d tmp;
                bool dst_closer = rangeSuitable(src[e_id - 1], src[e_id], dst_b, obs, tmp);
                if (dst_closer == true) return;        // 投影边的range更大
            }
            if (f_id < 0 || e_id < 0) return;
            dst.valid = false;
            return;
        }
    } else {
        // 大小角逻辑
        const Eigen::Vector2d o2s = dst.back().block<2, 1>(0, 0) - obs, o2f = dst.front().block<2, 1>(0, 0) - obs;
        if (can_proj.front() == true) {         // 小角度（逆时针）
            const Eigen::Vector2d src_o2f = src.front().block<2, 1>(0, 0) - obs;
            if (src_o2f.dot(o2f) <= 0.0 && src_o2f.dot(o2s) <= 0.0)        // 反向光线
                return;
            const Eigen::Vector2d f2f = src.front().block<2, 1>(0, 0) - dst.front().block<2, 1>(0, 0),
                    f2s = src.back().block<2, 1>(0, 0) - dst.front().block<2, 1>(0, 0),
                    s2f = src.front().block<2, 1>(0, 0) - dst.back().block<2, 1>(0, 0);
            const Eigen::Vector2d nf2f(-f2f(1), f2f(0)), ns2f(-s2f(1), s2f(0)), nf2s(-f2s(1), f2s(0));
            if (nf2f.dot(o2f) >= 0) {           // src.first 超过 dst.first 可全覆盖
                if (nf2s.dot(o2f) < 0) {        // src.second 不超过 dst.first 真全覆盖
                    int s_id = src.rotatedBinarySearch(dst.front().z());
                    if (s_id > 0) {
                        Eigen::Vector3d intersect = Eigen::Vector3d::Zero();
                        bool dst_closer = rangeSuitable(src[s_id - 1], src[s_id], o2f, obs, intersect);
                        if (dst_closer == false)
                            dst.valid = false;
                    }
                }
                return;
            } else if (ns2f.dot(o2s) < 0) {     // src.first 超过 dst.second
                return;
            }
            beam = src[first_id].block<2, 1>(0, 0) - obs;
            angle = src[first_id].z();
        } else if (can_proj.back() == true){
            const Eigen::Vector2d src_o2s = src.back().block<2, 1>(0, 0) - obs;
            if (src_o2s.dot(o2s) <= 0.0 && src_o2s.dot(o2f) <= 0.0)        // 反向光线
                return;
            const Eigen::Vector2d s2s = src.back().block<2, 1>(0, 0) - dst.back().block<2, 1>(0, 0),
                    s2f = src.front().block<2, 1>(0, 0) - dst.back().block<2, 1>(0, 0),
                    f2s = src.back().block<2, 1>(0, 0) - dst.front().block<2, 1>(0, 0);
            const Eigen::Vector2d ns2s(-s2s(1), s2s(0)), ns2f(-s2f(1), s2f(0)), nf2s(-f2s(1), f2s(0));        // 求法向量
            if (ns2s.dot(o2s) < 0) {            // src.second 超过 dst.second 可能全覆盖
                if (ns2f.dot(o2s) >= 0) {        // src.first 不超过 dst.second 说明真全覆盖
                    int e_id = src.rotatedBinarySearch(dst.back().z());
                    if (e_id > 0) {
                        Eigen::Vector3d intersect = Eigen::Vector3d::Zero();
                        bool dst_closer = rangeSuitable(src[e_id - 1], src[e_id], o2s, obs, intersect);
                        if (dst_closer == false)
                            dst.valid = false;
                    }
                } 
                return;
            } else if (nf2s.dot(o2f) >= 0) {    // src.second 不超过 dst.first 完全无关
                return;
            }
            // 超界问题
            beam = src[second_id].block<2, 1>(0, 0) - obs;
            angle = src[second_id].z();
            pop_back_flag = false;          // pop_front
        } else return;              // 完全没有关联的两个edges     
    }
    int id = dst.rotatedBinarySearch(angle);
    assert(id > 0);
    Eigen::Vector3d intersect = Eigen::Vector3d::Zero();
    if (rangeSuitable(dst[id - 1], dst[id], beam, obs, intersect) == false) return;
    if (pop_back_flag == true) {            // 删除大角度
        int delete_cnt = static_cast<int>(dst.size()) - id;
        for (int i = 0; i < delete_cnt; i++)
            dst.pop_back();
        dst.emplace_back(intersect);
        dst.proj_ids.second = -1;           // 大角度删除
    } else {
        for (int i = 0; i < id; i++) {
            dst.pop_front();
        }
        dst.emplace_front(intersect);
        dst.proj_ids.first = -1;            // 小角度删除
        if (dst.proj_ids.second >= 0)       // 尾部id发生变化
            dst.proj_ids.second = static_cast<int>(dst.size()) - 1;
    }
    for (const Eigen::Vector3d& pt: dst) {
        double dist = (pt.block<2, 1>(0, 0) - obs).norm();
        if (dist < dst.min_dist) dst.min_dist = dist;
    }
}

void Object::breakEdge(Eigen::Vector2d b1, Eigen::Vector2d b2, Eigen::Vector2d obs, Edge& dst, HeapType& heap) {
    std::array<Eigen::Vector2d, 2> task = {b1, b2};
    std::vector<Eigen::Vector3d> crs;
    std::array<size_t, 2> ids = {0, 0};
    for (size_t i = 0; i < 2; i++) {
        const Eigen::Vector2d& beam = task[i];
        double this_angle = atan2(beam(1), beam(0));
        int id = dst.rotatedBinarySearch(this_angle);
        Eigen::Vector3d intersect = Eigen::Vector3d::Zero();
        if (rangeSuitable(dst[id - 1], dst[id], beam, obs, intersect) == false) return;
        crs.emplace_back(intersect);
        ids[i] = dst.size() - static_cast<size_t>(id);
    }
    Edge new_edge;
    size_t add_cnt = 0;
    for (Edge::const_reverse_iterator rit = dst.crbegin(); rit != dst.crend() && add_cnt < ids[1]; rit++, add_cnt++)
        new_edge.push_front(*rit);
    new_edge.push_front(crs[1]);
    new_edge.initWithObs(obs, dst.proj_ids.second);
    for (size_t i = 0; i < ids[0]; i++)
        dst.pop_back();
    dst.push_back(crs[0]);
    dst.proj_ids.second =  -1;
    for (const Eigen::Vector3d& pt: dst) {
        double dist = (pt.block<2, 1>(0, 0) - obs).norm();
        if (dist < dst.min_dist) dst.min_dist = dist;
    }
    edges.push_back(new_edge);
    heap.emplace(edges.size() - 1);              // 可能不太安全
}

void Object::visualizeEdges(cv::Mat& src, cv::Point obs) const{
    if (valid == false)
        return;
    int cnt = -1;
    for (const Edge& eg: edges) {
        cnt++;
        if (eg.valid == false) continue;
        for (size_t i = 1; i < eg.size(); i++) {
            cv::line(src, cv::Point(eg[i - 1].x(), eg[i - 1].y()), cv::Point(eg[i].x(), eg[i].y()), cv::Scalar(0, 255, 255), 3);
        }
        char str[8];
        snprintf(str, 8, "%d:%lu", cnt, eg.size());
		cv::putText(src, str, cv::Point(eg.front().x(), eg.front().y()) + cv::Point(10, 10),
					cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
        cv::circle(src, cv::Point(eg.front().x(), eg.front().y()), 3, cv::Scalar(255, 255, 0), -1);
        cv::circle(src, cv::Point(eg.back().x(), eg.back().y()), 3, cv::Scalar(255, 0, 255), -1);
    }
    cv::circle(src, obs, 4, cv::Scalar(0, 255, 0), -1);
}

Eigen::Vector3d Object::getIntersection(
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

bool Object::rangeSuitable(
    const Eigen::Vector3d& p1, 
    const Eigen::Vector3d& p2, 
    const Eigen::Vector2d& beam, 
    const Eigen::Vector2d& obs,
    Eigen::Vector3d& intersect
) const {
    intersect = getIntersection(beam, p1, p2, obs);
    double range = (intersect.block<2, 1>(0, 0) - obs).norm();
    return beam.norm() < range;
}
