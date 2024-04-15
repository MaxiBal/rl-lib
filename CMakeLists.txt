cmake_minimum_required(VERSION 3.10)

project(RLLib C)


set(SOURCE_FILES src/net.c)


# Build library
add_library(RLLib ${SOURCE_FILES})
target_include_directories(RLLib PUBLIC include/)
target_link_libraries(RLLib m)


# Tests

add_executable(InitializeNetworkTest tests/initialize_network.c)
target_link_libraries(InitializeNetworkTest LINK_PUBLIC RLLib)
add_test(NAME InitializeNetworkTest COMMAND InitializeNetworkTest)

add_executable(LSRL tests/lsrl.c)
target_link_libraries(LSRL LINK_PUBLIC RLLib)
add_test(NAME LSRL COMMAND LSRL)

add_executable(XORNetwork tests/xor_network.c)
target_link_libraries(XORNetwork LINK_PUBLIC RLLib)
add_test(NAME XORNetwork COMMAND XORNetwork)

add_executable(MNIST tests/mnist.c)
target_link_libraries(MNIST LINK_PUBLIC RLLib)
add_test(NAME MNIST COMMAND MNIST)