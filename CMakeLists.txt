
cmake_minimum_required (VERSION 2.6)
project (GeoSay)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
AUX_SOURCE_DIRECTORY(./src SRC_LIST)

# message(${SRC_LIST})

add_executable(GeoSay ${SRC_LIST})
target_link_libraries(GeoSay ${OpenCV_LIBS})