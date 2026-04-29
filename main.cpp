#include <iostream>
#include "include/tensor.hpp"
#include "include/allreduce.hpp"
#include "include/mock_communicator.hpp"
#include <thread>
using namespace std;

int main() {
    int world_size = 4;

    std::vector<std::shared_ptr<Channel>> channels(world_size);
    for (auto& ch : channels) ch = std::make_shared<Channel>();

    auto barrier_count = std::make_shared<std::atomic<int>>(0);
    auto barrier_mtx   = std::make_shared<std::mutex>();
    auto barrier_cv    = std::make_shared<std::condition_variable>();

    std::vector<Tensor> tensors;
    for (int i = 0; i < world_size; i++) {
        tensors.emplace_back(4);
        tensors[i].data() = {1.0f, 2.0f, 3.0f, 4.0f};
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < world_size; i++) {
        threads.emplace_back([&, i]() {
            MockCommunicator comm(i, world_size, channels, barrier_count, barrier_mtx, barrier_cv);
            allreduce(tensors[i], comm, i, world_size);
        });
    }
    for (auto& th : threads) th.join();

    for (float v : tensors[0].data()) std::cout << v << "\n";
}