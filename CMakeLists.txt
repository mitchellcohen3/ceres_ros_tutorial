cmake_minimum_required(VERSION 3.5)
project(ceres_ros_tutorial)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

SET(GCC_COVERAGE_LINK_FLAGS "-lstdc++fs")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
)

catkin_package(
  CATKIN_DEPENDS roscpp
  LIBRARIES ceres_tutorials_lib
  INCLUDE_DIRS src/
)

# Include our header files
include_directories(
    src
    ${EIGEN3_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
    ${CERES_LIBRARIES}
    ${catkin_LIBRARIES}
)

# Make the shared library
list(APPEND LIBRARY_SOURCES 
    src/ceres/PoseLocalParameterization.cpp
    src/ceres/RelativeLandmarkFactor.cpp
    src/utils/SE3.cpp
    src/utils/SO3.cpp
    src/utils/Visualization.cpp
)

add_library(ceres_tutorials SHARED ${LIBRARY_SOURCES})
target_link_libraries(ceres_tutorials Eigen3::Eigen ${thirdparty_libraries})
target_include_directories(ceres_tutorials PUBLIC src/)
install(TARGETS ceres_tutorials
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Make the binary files
add_executable(simple_ba_example src/simple_ba_example.cpp)
target_link_libraries(simple_ba_example ceres_tutorials)