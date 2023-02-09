cmake_minimum_required (VERSION 3.14 FATAL_ERROR)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Compiler flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -g -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-DNEDEBUG -O3")

project(Waves2AMR)

# Target names
set(exe_name "Waves2AMR")
set(exe_utest_name "${exe_name}_unit_tests")
set(exe_lib "w2a_obj")

# Main executable (for now)
add_executable(${exe_name} main.cpp read_modes.cpp read_ascii.cpp)
# Communicate source directory to code
target_compile_definitions(${exe_name} PRIVATE src_dir="${CMAKE_SOURCE_DIR}")
# Make object files accessible (for sake of gtest code)
add_library(${exe_lib} OBJECT main.cpp read_modes.cpp read_ascii.cpp)
target_compile_definitions(${exe_lib} PRIVATE src_dir="${CMAKE_SOURCE_DIR}")

option(ENABLE_UNIT_TESTS "Enable unit testing" OFF)
option(ENABLE_TESTS "Enable regression testing" OFF)

if(ENABLE_UNIT_TESTS)
  add_executable(${exe_utest_name})
  add_subdirectory("googletest")
  add_subdirectory("unit_tests")
endif()

if(ENABLE_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory("tests")
endif()
