target_include_directories(
   ${w2a_lib_name}
   PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
   )

target_sources(
   ${w2a_lib_name}
   PRIVATE
   read_modes.h
   read_modes.cpp
   read_ascii.cpp
   modes_hosgrid.h
   modes_hosgrid.cpp
   interp_to_mfab.h
   interp_to_mfab.cpp
   )