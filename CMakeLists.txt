## This executable is just for making sure the setup can compile and run
cmake_minimum_required (VERSION 3.14 FATAL_ERROR)

# Compiler flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -g -pedantic -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-DNEDEBUG -O3 -std=c++17")

project(Waves2AMR CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(WAVES2AMR_USE_INTERNAL_AMREX "Add AMReX as subproject" ON)
option(WAVES2AMR_ENABLE_MPI "Enable MPI" OFF)
option(WAVES2AMR_ENABLE_OPENMP "Enable OpenMP" OFF)
option(WAVES2AMR_ENABLE_FCOMPARE "Enable FCompare" OFF)
option(WAVES2AMR_ENABLE_CUDA "Enable CUDA" OFF)
option(WAVES2AMR_ENABLE_ROCM "Enable ROCm/HIP" OFF)
option(WAVES2AMR_ENABLE_SYCL "Enable Intel OneAPI SyCL" OFF)
option(WAVES2AMR_ENABLE_TINY_PROFILE "Enable AMReX TinyProfile support" OFF)
option(WAVES2AMR_ENABLE_UNIT_TESTS "Enable unit testing" ON)

# Target names
set(w2a_base "Waves2AMR")
set(w2a_lib_name "${w2a_base}_obj")
set(w2a_test_name "${w2a_base}_test")
set(w2a_utests_name "${w2a_base}_unit_tests")

# FFTW
set(fftw_lib ${CMAKE_CURRENT_SOURCE_DIR}/fftw/lib/libfftw3.a)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fftw/include/)

# Access source files
FILE(GLOB cppsrc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Object library for linking
add_library(${w2a_lib_name} OBJECT)
# Access source files and catch-all header
add_subdirectory(src)
add_subdirectory(include)
target_link_libraries(${w2a_lib_name} PRIVATE ${fftw_lib})
install(TARGETS ${w2a_lib_name} DESTINATION ${PROJECT_NAME})

# Main executable (for testing compilation and running)
add_executable(${w2a_test_name})
add_subdirectory(tests)
target_link_libraries(${w2a_test_name} PRIVATE ${fftw_lib})
target_link_libraries(${w2a_test_name} PUBLIC ${w2a_lib_name})

include(utils.cmake)
if (WAVES2AMR_ENABLE_CUDA)
  enable_language(CUDA)
  set(w2atargets "${w2a_lib_name};${w2a_test_name}")
  foreach(tgt IN LISTS w2atargets)
    set_cuda_build_properties(${tgt})
  endforeach()
endif()

# AMReX
init_amrex()
target_link_libraries(${w2a_lib_name} PUBLIC AMReX::amrex)

if(WAVES2AMR_ENABLE_UNIT_TESTS)
  add_executable(${w2a_utests_name})
  add_subdirectory("submods/googletest")
  add_subdirectory("unit_tests")
  set_cuda_build_properties(${w2a_utests_name})
endif()
