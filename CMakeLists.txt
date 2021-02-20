# set(the_description "3D point features")
# ocv_define_module(surface_matching opencv_core opencv_flann WRAP python)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(surface_matching)

# SET(PCL_INCLUDE_DIRS /usr/lib/x86_64-linux-gnu)
# SET(PLATFORM_SPECIFIC_LIBS "-lboost_system" "-lboost_thread" "-lpthread" "-lboost_date_time")
# # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
# set(CMAKE_THREAD_LIBS_INIT "-lpthread")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# set(CMAKE_HAVE_THREADS_LIBRARY 1)
# set(CMAKE_USE_WIN32_THREADS_INIT 0)
# set(CMAKE_USE_PTHREADS_INIT 1)
# set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_STANDARD 14)
# add_compile_options(-std=c++11)

#Pcl
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${OpenCV_DEFINITIONS} ${Boost_DEFINITIONS}) 

add_library(icp src/icp.cpp)
target_link_libraries(icp ${OpenCV_LIBRARIES})

add_library(pose_3d src/pose_3d.cpp)
target_link_libraries(pose_3d ${OpenCV_LIBRARIES})

add_library(ppf_helpers src/ppf_helpers.cpp)
target_link_libraries(ppf_helpers ${OpenCV_LIBRARIES})

add_library(t_hash_int src/t_hash_int.cpp)
target_link_libraries(t_hash_int ${OpenCV_LIBRARIES})

add_library(ppf_match_3d src/ppf_match_3d.cpp)
target_link_libraries(ppf_match_3d ${OpenCV_LIBRARIES} pose_3d t_hash_int)

add_executable (ppf_matcher ppf_matcher.cpp)
target_link_libraries (ppf_matcher ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ppf_match_3d ppf_helpers icp)

add_executable (model_normal_estimation model_normal_estimation.cpp)
target_link_libraries (model_normal_estimation ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ppf_helpers)

add_executable (scene_normal_estimation scene_normal_estimation.cpp)
target_link_libraries (scene_normal_estimation ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ppf_helpers)