#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include "tensor.hpp"

struct Channel {
    std::queue<Tensor> queue;
    std::mutex mtx;
    std::condition_variable cv;
};
