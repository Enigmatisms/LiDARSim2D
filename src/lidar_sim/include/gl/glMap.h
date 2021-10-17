/// @author (Qianyue He:https://github.com/Enigmatisms) @copyright Enigmatisms
#pragma once
#include <memory>
#include <vector>
#include <GL/glut.h>
#include <Eigen/Dense>
#include "utils/consts.h"

extern std::unique_ptr<Walls> wall_ptr;

void tesselation();

/// @brief 绘制点 封闭障碍物 正在绘制的障碍
void mapViz();

/// @brief 矩形模式绘制矩形 圆形模式绘制圆形
void specialViz();