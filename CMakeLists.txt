cmake_minimum_required(VERSION 3.12)
project(IHCPOptimization CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ihcp_optimizer
    src/main.cpp 
    src/pso.cpp
    src/ihcp_model.cpp
    src/run_experiment.cpp
    src/sga.cpp
    src/csv_logger.cpp)

target_include_directories(ihcp_optimizer PRIVATE include)