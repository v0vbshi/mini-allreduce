cmake_minimum_required(VERSION 3.16)
project(mini-allreduce)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mini-allreduce main.cpp src/allreduce.cpp src/mock_communicator.cpp)

find_package(Threads REQUIRED)
target_link_libraries(mini-allreduce PRIVATE Threads::Threads)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_allreduce tests/test_allreduce.cpp src/allreduce.cpp src/mock_communicator.cpp)
target_link_libraries(test_allreduce PRIVATE GTest::gtest_main Threads::Threads)
include(GoogleTest)
gtest_discover_tests(test_allreduce)