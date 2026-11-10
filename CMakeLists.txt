cmake_minimum_required(VERSION 3.12)
project(IHCPOptimization CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ihcp_optimizer
    src/main.cpp 
    src/optimizers/pso.cpp
    src/ihcp_model.cpp
    src/run_experiment.cpp
    "src/optimizers/rcga.cpp"
    src/csv_logger.cpp
    src/optimizers/cs.cpp
    src/optimizers/optimizer_factory.cpp
    src/optimizers/aco.cpp "include/optimizers/gwo.h" "src/optimizers/gwo.cpp" "include/random.h" "src/random.cpp")

target_include_directories(ihcp_optimizer PRIVATE include)

if(MSVC)
    target_compile_options(ihcp_optimizer PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(ihcp_optimizer PRIVATE -O3 -ffast-math -march=native)
endif()