add_library(point_detection STATIC)

# Conditionally select implementation based on CUDA support
if(WITH_CUDA)
    set(EXAMPLE_IMPL src/example.cu)
    set(CUDA_DEPS CUDA::cudart)
else()
    set(EXAMPLE_IMPL src/example.cpp)
    set(CUDA_DEPS "")
endif()

target_sources(
    point_detection
    PRIVATE
        ${EXAMPLE_IMPL}
    PUBLIC
        include/point_detection/example.hpp
)

if(WITH_CUDA)
    set_target_properties(point_detection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

target_include_directories(
    point_detection
    PUBLIC
        include
)

target_link_libraries(
    point_detection
    PRIVATE
        ${CUDA_DEPS}
)


add_executable(hello src/hello_main.cpp)

target_link_libraries(hello PRIVATE point_detection)