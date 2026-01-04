cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

project(Hello LANGUAGES CXX)

# Handle CUDA support
option(WITH_CUDA "Build with CUDA support" OFF)
if(WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "CUDA Architectures not specified. Defaulting to 'native'.")
    else()
        message(STATUS "Building for CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Use system's CUDA runtime library
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    
    find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(src)