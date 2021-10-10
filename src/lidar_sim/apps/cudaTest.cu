#include "cuda_pf.hpp"

int main(int argc, char** argv) {
    cv::Mat occ(900, 1200, CV_8UC1);
    Eigen::Vector3d angles(1.5, -1.5, 0.0017);
    CudaPF cpf(occ, angles, 4000);
    return 0;
}
