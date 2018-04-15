
cmake_minimum_required(VERSION 2.8)
 
project(HW)
 
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(HW MACOSX_BUNDLE HW.cpp)
target_link_libraries(HW  ${ITK_LIBRARIES})
