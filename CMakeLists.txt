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
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp") 

#Pcl
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) # 二进制文件的输出目录
link_directories(${PROJECT_SOURCE_DIR}/lib/yaml-cpp)
# add_executable(${PROJECT_NAME} main.cpp)
# target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include) 

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include include/vcglib)
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
target_link_libraries (ppf_matcher ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ppf_match_3d ppf_helpers icp yaml-cpp.a)

# add_executable (model_normal_estimation model_normal_estimation.cpp)
# target_link_libraries (model_normal_estimation ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ppf_helpers)

# add_executable (scene_normal_estimation scene_normal_estimation.cpp)
# target_link_libraries (scene_normal_estimation ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  ppf_helpers)

# add_executable (segment_cluster segment_cluster.cpp)
# target_link_libraries (segment_cluster ${PCL_LIBRARIES})

# add_executable (test_mat test.cpp)
# target_link_libraries (test_mat ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

# add_executable(poisson_disk_sampling poisson_disk_sampling.cpp)
# target_link_libraries (poisson_disk_sampling ${PCL_LIBRARIES})

# add_executable (convert_mm_to_m convert_mm_to_m.cpp)
# target_link_libraries (convert_mm_to_m ${PCL_LIBRARIES})

add_executable (linemod_normal linemod_normal.cpp)
target_link_libraries (linemod_normal ${OpenCV_LIBRARIES} ppf_helpers)