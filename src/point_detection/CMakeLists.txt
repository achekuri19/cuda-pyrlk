add_library(point_detection)

target_sources(
    point_detection
    PRIVATE
        src/example.cpp
    PUBLIC
        include/example.hpp
)

target_include_directories(
    point_detection
    PUBLIC
        include
)


add_executable(hello src/hello_main.cpp)

target_link_libraries(hello PRIVATE point_detection)
target_link_options(hello PRIVATE -static)