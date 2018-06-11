cmake_minimum_required(VERSION 2.8)
project( squeezeCNN )
find_package(OpenCV REQUIRED )
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( squeezecnn mainAlgo.cpp face_extraction.cpp training_generator.cpp )

target_link_libraries( squeezecnn ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( squeezecnn LINK_PUBLIC ${Boost_LIBRARIES} )

set(CMAKE_BUILD_TYPE Debug)
