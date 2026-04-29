cmake_minimum_required(VERSION 3.16)
project(mini-allreduce)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMM_SRCS src/allreduce.cpp src/mock_communicator.cpp src/efa_communicator.cpp)

add_executable(mini-allreduce main.cpp ${COMM_SRCS})

find_package(Threads REQUIRED)
target_link_libraries(mini-allreduce PRIVATE Threads::Threads)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_allreduce tests/test_allreduce.cpp ${COMM_SRCS})
target_link_libraries(test_allreduce PRIVATE GTest::gtest_main Threads::Threads)
include(GoogleTest)
gtest_discover_tests(test_allreduce)

add_executable(bench_allreduce benchmarks/bench_allreduce.cpp ${COMM_SRCS})
target_link_libraries(bench_allreduce PRIVATE Threads::Threads)