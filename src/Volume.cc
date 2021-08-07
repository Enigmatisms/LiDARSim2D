#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Volume.hpp"
#include "consts.h"
#include "LOG.hpp"

const cv::Rect walls(0, 0, 1200, 900);
const cv::Rect floors(30, 30, 1140, 840);
const double luminous_range = 2.5e5;

void Volume::visualizeVisualSpace(const std::vector<Obstacle>& _obstcs, const Eigen::Vector2d& obs, cv::Mat& dst) const {
    cv::rectangle(dst, walls, cv::Scalar(10, 10, 10), -1);
    cv::rectangle(dst, floors, cv::Scalar(40, 40, 40), -1);
    std::vector<std::vector<cv::Point>> polygons;
    for (const Object& obj: objs)
        obj.makePolygons4Render(obs, polygons);
    cv::drawContours(dst, polygons, -1, cv::Scalar(254, 254, 254), -1);
    cv::drawContours(dst, _obstcs, -1, cv::Scalar(10, 10, 10), -1);
    prettier(dst, obs);
    cv::circle(dst, cv::Point(obs.x(), obs.y()), 4, cv::Scalar(0, 255, 255), -1);
    cv::circle(dst, cv::Point(obs.x(), obs.y()), 7, cv::Scalar(40, 40, 40), 2);
}

void Volume::prettier(cv::Mat& src, const Eigen::Vector2d& obs) const {
    src.forEach<cv::Vec3b>(
        [&](cv::Vec3b& pix, const int* pos){
            if (pix[0] > 128){        // 为白色
                double dist = std::pow(pos[1] - obs.x(), 2) + std::pow(pos[0] - obs.y(), 2);
                uchar res = std::max(40.0, 254.0 - 214.0 / luminous_range * dist);
                pix[0] = pix[1] = pix[2] = res;
            }
        }
    );
}


void Volume::calculateVisualSpace(const std::vector<Obstacle>& _obstcs, cv::Point obs, cv::Mat& src) {
    objs.clear();
    Eigen::Vector2d observing_point(obs.x, obs.y); 
    std::vector<std::vector<Eigen::Vector2d>> obstcs;
    for (const Obstacle& obstacle: _obstcs) {            // 构建objects
        obstcs.emplace_back();
        for (const cv::Point& pt: obstacle)
            obstcs.back().emplace_back(pt.x, pt.y);
    }
    // ================ add walls ================
    obstcs.emplace_back();
    for (const cv::Point& pt: north_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 
    obstcs.emplace_back();
    for (const cv::Point& pt: east_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 
    obstcs.emplace_back();
    for (const cv::Point& pt: south_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 
    obstcs.emplace_back();
    for (const cv::Point& pt: west_wall)
        obstcs.back().emplace_back(pt.x, pt.y); 

    int obj_cnt = 0;
    for (const std::vector<Eigen::Vector2d>& obstacle: obstcs) {            // 构建objects
        Object obj(obj_cnt);
        obj.intialize(obstacle, observing_point);
        objs.push_back(obj);
        obj_cnt++;
    }
    for (size_t i = 0; i < objs.size(); i++)            // 构建堆
        heap.emplace(i);
    while (heap.empty() == false) {
        size_t obj_id = heap.top();
        Object& obj = objs[obj_id];
        heap.pop();
        LOG_ERROR("Object %d, started to internal project.", obj_id);
        obj.internalProjection(observing_point);
        LOG_MARK("After interal proj, valids in object %d are:", obj_id);
        LOG_MARK("Object %d, started to external project.", obj_id);
        for (Object& projectee: objs) {
            // 不查看完全被遮挡的，不投影已经投影过的
            if (projectee.valid == false || obj.id == projectee.id) continue;
            LOG_GAY("External occ, object %lu", projectee.id);
            // 选择projectee被投影
            // if (obj_id == 11 && projectee.id == 17)
            //     simplePreVisualize(src, obs);
            obj.externalOcclusion(projectee, observing_point);
            LOG_CHECK("Projectee (%d) processed, edges (valid):", projectee.id);
            for (Edge& eg: projectee.edges) {
                if (eg.valid == false || eg.size() > 2) continue;
                Eigen::Vector2d diff = eg.front().block<2, 1>(0, 0) - eg.back().block<2, 1>(0, 0);
                if (diff.norm() < 1.0) {
                    LOG_ERROR("Projectee %d has one singular edge, (%.4lf, %.4lf), (%.4lf, %.4lf)", projectee.id, eg.front().x(), eg.front().y(), eg.back().x(), eg.back().y());
                    eg.valid = false;
                }
            }
        }
        LOG_SHELL("After external proj, valids in object %d are:", obj_id);
    }
}

void Volume::simplePreVisualize(cv::Mat& src, const cv::Point& obs) const {
    for (const Object& obj: objs) {
        obj.visualizeEdges(src, obs);
        char str[8];
        snprintf(str, 8, "%d", obj.id);
		cv::putText(src, str, cv::Point(obj.edges.front().back().x(), obj.edges.front().back().y()) + cv::Point(10, 10),
				cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 255));
    }
    cv::imshow("tmp", src);
    cv::waitKey(0);
}
