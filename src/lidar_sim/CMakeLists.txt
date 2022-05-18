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
      -gencode=arch=compute_86,code=sm_86
      -lcudadevrt -lcudart
      --expt-relaxed-constexpr
      -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
  )
  add_compile_options(-Wno-deprecated-declarations)

  find_package(Boost REQUIRED COMPONENTS
    filesystem iostreams
  )
  if(NOT Boost_FOUND)
     message(FATAL_ERROR "Boost not found.")
  endif()
  if (CUDA_VERSION VERSION_LESS "10.2")
    find_package(Eigen3 REQUIRED)
    if(NOT Eigen3_FOUND)
      message(FATAL_ERROR "Eigen3 not found.")
    endif()
  else()
    set(EIGEN3_INCLUDE_DIRS "/home/stn/extra_modules/eigen-3.4.0/")
  endif()
else()
  find_package(Eigen3 REQUIRED)
  if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found.")
  endif()
  message(STATUS ${Eigen3_VERSION})
endif()

# set(OpenGL_GL_PREFERENCE GLVND)
# find_package(OpenGL REQUIRED)
# if(NOT OpenGL_FOUND)
#   message(FATAL_ERROR "OpenGL not found.")
# endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS nav_msgs rosbag roscpp sensor_msgs std_msgs tf visualization_msgs
)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenGL_INCLUDE_DIR}
)

aux_source_directory(src/utils UTILS_SRC)
aux_source_directory(src/cuda CUDA_SRC)
aux_source_directory(src/volume VOL_SRC)
# aux_source_directory(src/gl GL_SRC)

add_executable(filter apps/filter.cc
  ${VOL_SRC}
  ${UTILS_SRC}
)

target_link_libraries(filter
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
)

add_executable(scan apps/scan_sim.cc
  ${VOL_SRC}
  ${UTILS_SRC}
)

target_link_libraries(scan
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
  stdc++fs
)

add_executable(scan_joy apps/scan_sim_joy.cc
  ${VOL_SRC}
  ${UTILS_SRC}
)

target_link_libraries(scan_joy
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
)

add_executable(editor apps/mapEditor.cc
  ${UTILS_SRC}
)

target_link_libraries(editor
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  pthread
)

add_executable(test_key apps/testKey.cc
  src/utils/keyCtrl.cc
)

target_link_libraries(test_key
  ${catkin_LIBRARIES}
  pthread
)

add_executable(test_joy apps/testJoy.cc
  src/utils/joyCtrl.cc
)

target_link_libraries(test_joy
  ${catkin_LIBRARIES}
  pthread
)

install(TARGETS filter scan editor test_key test_joy scan_joy
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

if (COMPILE_CUDA)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.3/bin")
  add_compile_options(-Wno-deprecated-declarations)
  cuda_add_executable(cuda_test
    apps/cudaTest.cu
    src/utils/consts.cc
    src/utils/mapEdit.cc
    src/utils/keyCtrl.cc
    src/cuda/cuda_funcs.cu
    src/cuda/cuda_pf.cu
  )
  enable_language(CUDA)
  set_property(TARGET cuda_test PROPERTY CXX_STANDARD 14)

  target_link_libraries(cuda_test
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
  )

  cuda_add_executable(cuda_scan
    apps/scan_sim_cuda.cu
    src/utils/consts.cc
    src/utils/mapEdit.cc
    src/utils/keyCtrl.cc
    src/utils/scanUtils.cc
    src/utils/vizUtils.cc
    src/cuda/ray_tracer.cu
    src/cuda/host_func.cu
  )
  enable_language(CUDA)
  set_property(TARGET cuda_scan PROPERTY CXX_STANDARD 14)

  target_link_libraries(cuda_scan
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
  )
endif()

# add_executable(gl_test apps/glMapEditor.cc
#   ${GL_SRC}
#   src/utils/consts.cc
#   src/utils/mapEdit.cc
# )

# target_link_libraries(gl_test
#   ${catkin_LIBRARIES}
#   ${OPENGL_LIBRARY}
#   ${OpenCV_LIBS}
#   glut
# )
