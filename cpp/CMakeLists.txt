cmake_minimum_required(VERSION 3.10)
project(vpi_demo)

set(CMAKE_CXX_STANDARD 11)

# 寻找VPI和OpenCV
find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(vpi_demo main.cpp)
# 链接库文件
target_link_libraries(vpi_demo vpi ${OpenCV_LIBS})
