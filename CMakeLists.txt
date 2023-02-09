## This executable is just for making sure the setup can compile and run
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
set(exe_base "Waves2AMR")
set(exe_name "${exe_base}_test")
set(exe_utest_name "${exe_base}_unit_tests")

# Main executable (for testing compilation and running)
add_executable(${exe_name} tests/main.cpp)

option(ENABLE_UNIT_TESTS "Enable unit testing" ON)

if(ENABLE_UNIT_TESTS)
  add_executable(${exe_utest_name})
  add_subdirectory("googletest")
  add_subdirectory("unit_tests")
endif()
