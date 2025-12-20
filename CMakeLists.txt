cmake_minimum_required(VERSION 3.24)
set(CMAKE_EXE_LINKER_FLAGS "-static")

project(Hello)
add_executable(hello hello_main.cpp)