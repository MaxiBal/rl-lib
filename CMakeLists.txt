cmake_minimum_required(VERSION 3.10)

project(RLLib C)


set(SOURCE_FILES src/net.c)


# Build library
add_library(RLLib ${SOURCE_FILES})
target_include_directories(RLLib PUBLIC include/)
target_link_libraries(RLLib m)


# Add main executable
add_executable(RLMain src/main.c)

target_link_libraries(RLMain LINK_PUBLIC RLLib)