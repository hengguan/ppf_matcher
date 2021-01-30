# set(the_description "3D point features")
# ocv_define_module(surface_matching opencv_core opencv_flann WRAP python)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(surface_matching)

SET(PCL_INCLUDE_DIRS /usr/lib/x86_64-linux-gnu)

#Pcl
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${OpenCV_DEFINITIONS}) 

add_library(ppf_match_3d src/ppf_match_3d.cpp)
target_link_libraries(ppf_match_3d ${OpenCV_LIBRARIES})

add_executable (ppf_matcher ppf_matcher.cpp)
target_link_libraries (ppf_matcher ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})