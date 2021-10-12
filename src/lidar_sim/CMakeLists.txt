cmake_minimum_required(VERSION 3.0.2)
project(lidar_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(COMPILE_CUDA True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  rosbag
  roscpp
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 3 not found.")
endif()

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
  message(FATAL_ERROR "Eigen3 not found.")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost REQUIRED COMPONENTS system)

if (COMPILE_CUDA)
  find_package(CUDA REQUIRED)
  set(
      CUDA_NVCC_FLAGS
      ${CUDA_NVCC_FLAGS};
      -O3 --use_fast_math
      -gencode=arch=compute_61,code=sm_61
      -lcudadevrt -lcudart
      --expt-relaxed-constexpr
  )
  add_compile_options(-Wno-deprecated-declarations)
  SET(CUDA_SEPARABLE_COMPILATION ON)

  find_package(Boost REQUIRED COMPONENTS
    filesystem iostreams
  )
  if(NOT Boost_FOUND)
     message(FATAL_ERROR "Boost not found.")
  endif()
endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS nav_msgs rosbag roscpp sensor_msgs std_msgs tf visualization_msgs
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

aux_source_directory(src/ SRC)

add_executable(filter apps/filter.cc
  ${SRC}
)

target_link_libraries(filter
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
)

add_executable(scan apps/scan_sim.cc
  ${SRC}
)

target_link_libraries(scan
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
)

add_executable(editor apps/mapEditor.cc
  ${SRC}
)

target_link_libraries(editor
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
)

add_executable(test_key apps/testKey.cc
  src/keyCtrl.cc
)

target_link_libraries(test_key
  ${catkin_LIBRARIES}
  pthread
)

install(TARGETS filter scan editor test_key
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

if (COMPILE_CUDA)
  aux_source_directory(cuda/ CUDA_SRC)
  cuda_add_executable(cuda_test
    apps/cudaTest.cu
    src/consts.cc
    src/mapEdit.cc
    src/keyCtrl.cc
    ${CUDA_SRC}
  )

  set_property(TARGET cuda_test PROPERTY CXX_STANDARD 14)

  target_link_libraries(cuda_test
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
    /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudadevrt.a
  )
endif()
