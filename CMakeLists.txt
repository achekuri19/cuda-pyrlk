cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

project(Hello LANGUAGES CXX CUDA)

# Enable CUDA language if requested
option(WITH_CUDA "Build with CUDA support" OFF)
if(WITH_CUDA)
    enable_language(CUDA)
endif()


add_executable(hello hello_main.cpp)
target_link_options(hello PRIVATE -static)