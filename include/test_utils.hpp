#pragma once
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include "tensor.hpp"
#include "allreduce.hpp"
#include "mock_communicator.hpp"
#include "efa_communicator.hpp"

inline void make_shared_state(int world_size,
    std::vector<std::shared_ptr<Channel>>& channels,
    std::shared_ptr<std::atomic<int>>& barrier_count,
    std::shared_ptr<std::mutex>& barrier_mtx,
    std::shared_ptr<std::condition_variable>& barrier_cv)
{
    channels.resize(world_size);
    for (auto& ch : channels) ch = std::make_shared<Channel>();
    barrier_count = std::make_shared<std::atomic<int>>(0);
    barrier_mtx   = std::make_shared<std::mutex>();
    barrier_cv    = std::make_shared<std::condition_variable>();
}

inline void run_allreduce(std::vector<Tensor>& tensors, int world_size) {
    std::vector<std::shared_ptr<Channel>> channels;
    std::shared_ptr<std::atomic<int>> barrier_count;
    std::shared_ptr<std::mutex> barrier_mtx;
    std::shared_ptr<std::condition_variable> barrier_cv;
    make_shared_state(world_size, channels, barrier_count, barrier_mtx, barrier_cv);

    std::vector<std::thread> threads;
    for (int r = 0; r < world_size; r++) {
        threads.emplace_back([&, r]() {
            MockCommunicator comm(r, world_size, channels, barrier_count, barrier_mtx, barrier_cv);
            allreduce(tensors[r], comm, r, world_size);
        });
    }
    for (auto& th : threads) th.join();
}

inline void run_efa_allreduce(std::vector<Tensor>& tensors, int world_size, int latency_us) {
    std::vector<std::shared_ptr<Channel>> channels;
    std::shared_ptr<std::atomic<int>> barrier_count;
    std::shared_ptr<std::mutex> barrier_mtx;
    std::shared_ptr<std::condition_variable> barrier_cv;
    make_shared_state(world_size, channels, barrier_count, barrier_mtx, barrier_cv);

    std::vector<std::thread> threads;
    for (int r = 0; r < world_size; r++) {
        threads.emplace_back([&, r]() {
            EfaCommunicator comm(r, world_size, latency_us, channels, barrier_count, barrier_mtx, barrier_cv);
            allreduce(tensors[r], comm, r, world_size);
        });
    }
    for (auto& th : threads) th.join();
}
