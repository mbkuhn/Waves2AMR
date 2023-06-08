## This executable is just for making sure the setup can compile and run
cmake_minimum_required (VERSION 3.14 FATAL_ERROR)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Compiler flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -g -pedantic -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-DNEDEBUG -O3 -std=c++17")

project(Waves2AMR)

# Options
option(WAVES2AMR_USE_INTERNAL_AMREX "Add AMReX as subproject" ON)
option(WAVES2AMR_ENABLE_MPI "Enable MPI" OFF)
option(WAVES2AMR_ENABLE_OPENMP "Enable OpenMP" OFF)
option(WAVES2AMR_ENABLE_FCOMPARE "Enable FCompare" OFF)
option(WAVES2AMR_ENABLE_CUDA "Enable CUDA" OFF)
option(WAVES2AMR_ENABLE_ROCM "Enable ROCm/HIP" OFF)
option(WAVES2AMR_ENABLE_SYCL "Enable Intel OneAPI SyCL" OFF)

# Target names
set(exe_base "Waves2AMR")
set(exe_name "${exe_base}_test")
set(exe_utest_name "${exe_base}_unit_tests")
set(exe_lib "${exe_base}_obj")

# FFTW
set(fftw_lib ${CMAKE_CURRENT_SOURCE_DIR}/fftw/lib/libfftw3.a)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fftw/include/)

# Access source files
FILE(GLOB cppsrc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Library for external use
add_library(${exe_base} SHARED ${cppsrc})
target_link_libraries(${exe_base} ${fftw_lib})
install(TARGETS ${exe_base} DESTINATION ${PROJECT_NAME})

# Make object files accessible (for sake of gtest code)
add_library(${exe_lib} OBJECT ${cppsrc})
target_compile_definitions(${exe_lib} PRIVATE src_dir="${CMAKE_SOURCE_DIR}")
target_link_libraries(${exe_lib} ${fftw_lib})

# Main executable (for testing compilation and running)
add_executable(${exe_name} ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cpp ${cppsrc})
target_link_libraries(${exe_name} ${fftw_lib})

# AMReX
include(amrex_utils.cmake)
init_amrex()
target_link_libraries(${exe_base} AMReX::amrex)
target_link_libraries(${exe_lib} AMReX::amrex)
target_link_libraries(${exe_name} AMReX::amrex)

option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

if(ENABLE_UNIT_TESTS)
  add_executable(${exe_utest_name})
  add_subdirectory("submods/googletest")
  add_subdirectory("unit_tests")
endif()
