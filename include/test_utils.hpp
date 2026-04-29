#pragma once
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include "tensor.hpp"
#include "allreduce.hpp"
#include "mock_communicator.hpp"

inline void run_allreduce(std::vector<Tensor>& tensors, int world_size) {
    auto channels = std::vector<std::shared_ptr<Channel>>(world_size);
    for (auto& ch : channels) ch = std::make_shared<Channel>();

    auto barrier_count = std::make_shared<std::atomic<int>>(0);
    auto barrier_mtx   = std::make_shared<std::mutex>();
    auto barrier_cv    = std::make_shared<std::condition_variable>();

    std::vector<std::thread> threads;
    for (int r = 0; r < world_size; r++) {
        threads.emplace_back([&, r]() {
            MockCommunicator comm(r, world_size, channels, barrier_count, barrier_mtx, barrier_cv);
            allreduce(tensors[r], comm, r, world_size);
        });
    }
    for (auto& th : threads) th.join();
}
