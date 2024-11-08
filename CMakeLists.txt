cmake_minimum_required(VERSION 3.14)
project(MedialAxisShrinkingBall)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(EXE_MASB masb)
add_executable(${EXE_MASB} ${SRC_DIR}/masb.cpp)
target_link_libraries(${EXE_MASB} PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX z cnpy)
target_include_directories(${EXE_MASB} PUBLIC ${INC_DIR})

set(EXE_XYZ2NPY xyz2npy)
add_executable(${EXE_XYZ2NPY} ${SRC_DIR}/xyz2npy.cpp)
target_link_libraries(${EXE_XYZ2NPY} PUBLIC z cnpy)
