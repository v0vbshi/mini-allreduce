cmake_minimum_required(VERSION 3.16)
project(mini-allreduce)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mini-allreduce main.cpp src/allreduce.cpp src/mock_communicator.cpp)

find_package(Threads REQUIRED)
target_link_libraries(mini-allreduce PRIVATE Threads::Threads)