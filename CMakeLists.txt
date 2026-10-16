cmake_minimum_required(VERSION 3.12)
project(IHCP_PSO CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ihcp_pso 
    src/main.cpp 
    "src/pso.cpp"
    src/ihcp_model.cpp "include/run_experiment.h" "include/experiment_result.h" "src/run_experiment.cpp" "include/ihcp_result.h" "include/scope_tracer.h" "include/sga.h" "include/ihcp_parameters.h" "src/sga.cpp")

target_include_directories(ihcp_pso PRIVATE include)