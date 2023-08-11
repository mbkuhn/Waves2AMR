target_sources(${w2a_utests_name}
               PRIVATE
               test_readops.cpp
               test_ascii.cpp
               test_fftexamples.cpp
               test_hosgrid.cpp
               test_combined.cpp
               test_AMReXexamples.cpp
               test_interp_to_mfab.cpp
               utest_main.cpp)
target_include_directories(${w2a_utests_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${w2a_utests_name} PRIVATE gtest)
target_link_libraries(${w2a_utests_name} PUBLIC ${w2a_lib_name})
target_include_directories(${w2a_utests_name} SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/submods/googletest/googletest/include)
