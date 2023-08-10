cmake_minimum_required(VERSION 3.14)

project( Waves2AMR
    LANGUAGES C CXX
    )

message(STATUS "CMake version: ${CMAKE_VERSION}")

#
# Check if CMAKE_BUILD_TYPE is given. If not, use default
#
if ( NOT CMAKE_BUILD_TYPE )
   set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
   set(CMAKE_BUILD_TYPE Release
       CACHE STRING
       "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
else ()
   message(STATUS "Build type set by user to '${CMAKE_BUILD_TYPE}'.")
endif()

#
# Options
#
option(WAVES2AMR_ENABLE_MPI "Enable MPI" OFF)

set(WAVES2AMR_GPU_BACKEND_VALUES NONE SYCL CUDA HIP)
set(WAVES2AMR_GPU_BACKEND NONE CACHE STRING "On-node, accelerated GPU backend: <NONE,SYCL,CUDA,HIP>")
set_property(CACHE WAVES2AMR_GPU_BACKEND PROPERTY STRINGS ${WAVES2AMR_GPU_BACKEND_VALUES})
if (NOT WAVES2AMR_GPU_BACKEND IN_LIST WAVES2AMR_GPU_BACKEND_VALUES)
   message(FATAL_ERROR "WAVES2AMR_GPU_BACKEND=${WAVES2AMR_GPU_BACKEND} is not allowed."
      " Must be one of ${WAVES2AMR_GPU_BACKEND_VALUES}")
endif ()

#
# Find AMReX, check for required components
#
if (NOT TARGET AMReX::amrex)
   set(AMREX_REQUIRED_COMPONENTS 3D DOUBLE)
   if (WAVES2AMR_ENABLE_MPI)
      list(APPEND AMREX_REQUIRED_COMPONENTS MPI)
   endif ()
   if (NOT WAVES2AMR_GPU_BACKEND STREQUAL "NONE")
      list(APPEND AMREX_REQUIRED_COMPONENTS ${WAVES2AMR_GPU_BACKEND})
   endif ()
   find_package(AMReX CONFIG REQUIRED ${AMREX_REQUIRED_COMPONENTS} )
endif ()

#
# Enable CUDA if requested
#
if (WAVES2AMR_GPU_BACKEND STREQUAL "CUDA")
   # CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
   # https://cmake.org/cmake/help/latest/policy/CMP0104.html
   if(POLICY CMP0104)
      cmake_policy(SET CMP0104 OLD)
   endif()
   include(AMReX_SetupCUDA)
   include(AMReXTargetHelpers)
endif ()

# FFTW library is required, but I need better sourcing
set(fftw_lib ${CMAKE_CURRENT_SOURCE_DIR}/fftw/lib/libfftw3.a)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/fftw/include/)

#
# Define the object library to compile
#
add_library(waves_2_amr OBJECT)
if (BUILD_SHARED_LIBS)
  set_target_properties(waves_2_amr PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
# Link required libraries
target_link_libraries(waves_2_amr PUBLIC AMReX::amrex)
target_link_libraries(waves_2_amr PRIVATE ${fftw_lib})
add_subdirectory(src)

if (WAVES2AMR_GPU_BACKEND STREQUAL "CUDA")
   setup_target_for_cuda_compilation(waves_2_amr)
endif ()

if ( NOT CMAKE_CXX_FLAGS )
   target_link_libraries(waves_2_amr PUBLIC AMReX::Flags_CXX)
endif ()


# Installation rules
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Create non-object library for use as external target
add_library(waves_2_amr_api)
if (BUILD_SHARED_LIBS)
  set_target_properties(waves_2_amr_api PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_link_libraries(waves_2_amr_api PUBLIC waves_2_amr)
add_library(${PROJECT_NAME}::waves_2_amr_api ALIAS waves_2_amr)

# Collect all headers and make them installable with the target
get_target_property(WAVES2AMR_INCLUDES waves_2_amr SOURCES)
list(FILTER WAVES2AMR_INCLUDES INCLUDE REGEX "\\.H")
set_target_properties(
  waves_2_amr PROPERTIES PUBLIC_HEADER "${WAVES2AMR_INCLUDES}")

# Install Waves2AMR
install(
  TARGETS waves_2_amr_api waves_2_amr
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME       DESTINATION bin
  ARCHIVE       DESTINATION lib
  LIBRARY       DESTINATION lib
  INCLUDES      DESTINATION include
  PUBLIC_HEADER DESTINATION include
  )

# Make Waves2AMR discoverable using `find_package`
install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )
