target_include_directories(
   waves_2_amr
   PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
   )

target_sources(
   waves_2_amr
   PRIVATE
   read_modes.h
   read_modes.cpp
   read_ascii.cpp
   modes_hosgrid.h
   modes_hosgrid.cpp
   interp_to_mfab.h
   interp_to_mfab.cpp
   )