target_sources(${exe_utest_name}
               PRIVATE
	           test_readops.cpp
               test_ascii.cpp
               ${CMAKE_SOURCE_DIR}/googletest/googletest/src/gtest_main.cc)
target_include_directories(${exe_utest_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${exe_utest_name} PRIVATE gtest)
target_link_libraries(${exe_utest_name} PUBLIC ${exe_lib})
target_include_directories(${exe_utest_name} SYSTEM PRIVATE
  ${CMAKE_SOURCE_DIR}/googletest/googletest/include)