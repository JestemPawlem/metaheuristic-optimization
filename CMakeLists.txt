cmake_minimum_required(VERSION 3.12)
project(IHCP_PSO CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ihcp_pso 
    src/main.cpp 
    src/pso.cpp 
    src/ihcp_model.cpp
 "include/color.h")

target_include_directories(ihcp_pso PRIVATE include)