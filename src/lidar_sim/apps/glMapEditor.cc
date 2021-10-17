#include "gl/viewer.hpp"
#include "gl/glMap.h"
#include "utils/mapEdit.h"

void makeMapBorder(Walls& walls) {
    Wall border;
    border.emplace_back(1200, 0, 0);
    border.emplace_back(0, 0, 0);
    border.emplace_back(0, 900, 0);
    border.emplace_back(1200, 900, 0);

    border.emplace_back(30, 30, 0);
    border.emplace_back(1170, 30, 0);
    border.emplace_back(1170, 870, 0);
    border.emplace_back(30, 870, 0);
    walls.push_back(border);
}

int main(int argc, char* argv[]) {
    Walls walls;
    mapLoad("/home/sentinel/ParticleFilter/maps/standard3.txt", walls);
    // makeMapBorder(walls);
    wall_ptr = std::unique_ptr<Walls>(&walls);
    printf("Size: %lu\n", wall_ptr->size());
    viewer(argc, argv);
    return 0;
}