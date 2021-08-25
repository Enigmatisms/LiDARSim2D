cmake_minimum_required(VERSION 2.8.3)
project(volume)
set(GDB_DEBUG FALSE)

## Torch 需要至少C++14进行编译
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

if (GDB_DEBUG)
   SET(CMAKE_BUILD_TYPE "Debug")
   SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
   SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
else()
   SET(CMAKE_BUILD_TYPE "Release")
endif()

link_directories(
  /usr/local/lib
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ./include/
)

aux_source_directory(src/ SRC)

add_executable(main main.cc
  ${SRC}
)

target_link_libraries(main
  ${OpenCV_LIBS}
  pthread
)

add_executable(filter filter.cc
  ${SRC}
)

target_link_libraries(filter
  ${OpenCV_LIBS}
  pthread
)


add_executable(scan scan_sim.cc
  ${SRC}
)

target_link_libraries(scan
  ${OpenCV_LIBS}
  pthread
)