#pragma once
#include "cast_kernel.hpp"

void deallocatePoints();

/** Allocate on intialization, 
 * global_memory (float* for angles, sizeof(float) * point_num)
 * global_memory (bool* for [mesh being valid], sizeof(bool) * point_num)
 * global_memory (char* for [next id], sizeof(char) * point_num)
 * constant memory for points
*/
void updatePointInfo(const Vec2* const meshes, const char* const nexts, int point_num, bool initialized);

// 所有点都已经在初始化时移动到常量内存中
void shadowCasting(const Vec3& pose, std::vector<Vec2>& host_output);
