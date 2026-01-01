add_library(point_detection)

# Conditionally select implementation based on CUDA support
if(WITH_CUDA)
    set(EXAMPLE_IMPL src/example.cu)
else()
    set(EXAMPLE_IMPL src/example.cpp)
endif()

target_sources(
    point_detection
    PRIVATE
        ${EXAMPLE_IMPL}
    PUBLIC
        include/point_detection/example.hpp
)

target_include_directories(
    point_detection
    PUBLIC
        include
)


add_executable(hello src/hello_main.cpp)

target_link_libraries(hello PRIVATE point_detection)