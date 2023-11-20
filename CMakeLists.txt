cmake_minimum_required(VERSION 3.15)

project(ORBDetector)

# C++ 17
set(CMAKE_CXX_STANDARD 17)

# Set compiler flags, these are the flags that will be used for all targets
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

# Append optimization flags to main CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# OpenCV
find_package(OpenCV REQUIRED)

# Boost
find_package(Boost REQUIRED)

# Source files
# Grep find all *.cpp files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add_executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Set the opencv include directory and the boost include directory
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Link the opencv libraries and the boost libraries and the hdf5 libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} /usr/local/lib/libboost_program_options.so)

# install the executable to the bin directory
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

