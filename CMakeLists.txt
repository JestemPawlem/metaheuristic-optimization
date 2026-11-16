cmake_minimum_required(VERSION 3.12)
project(IHCPOptimization CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ihcp_optimizer
    src/main.cpp "include/optimizers/aoa.h" "include/problems/benchmarks.h" "include/problems/problem_factory.h" "include/config.h")

target_include_directories(ihcp_optimizer PRIVATE include)

if(MSVC)
    target_compile_options(ihcp_optimizer PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(ihcp_optimizer PRIVATE -O3 -ffast-math -march=native)
endif()