#include <iostream>
#include "include/tensor.hpp"
#include "include/allreduce.hpp"
#include "include/mock_communicator.hpp"
#include <thread>
using namespace std;

// temp stub
// class StubComm : public Communicator {
// public:
//     void send(int, const Tensor&) override {}
//     void recv(int, Tensor&) override {}
//     void barrier() override {}
// };

int main() {
    int world_size = 4;

    // Step1: create shared channels
    vector<shared_ptr<Channel>> channels(world_size);
    for (auto& ch : channels) {
        ch = std::make_shared<Channel>();
    }
    // Step2: created shated barrier objects
    auto barrier_count = make_shared<atomic<int>>(0);
    auto barrier_mtx = make_shared<mutex>();
    auto barrier_cv = make_shared<condition_variable>();
    // Step3: create tensors for each rank
    vector<Tensor> tensors;
    for (int i = 0; i < world_size; i++) {
       tensors.emplace_back(4);
       tensors[i].data() = {1.0f, 2.0f, 3.0f, 4.0f};
    }
    // Step4: spwan threads, each running allreduce
    vector<thread> threads;
    for (int i = 0; i < world_size; i++) {
        threads.emplace_back([&, i]() {
            MockCommunicator comm(i, world_size, channels, barrier_count, barrier_mtx, barrier_cv);
            allreduce(tensors[i], comm, i, world_size);
        });
    }
    // Step5: join threads
    for (auto& th : threads) th.join();
    // Step6: print results
    for (float v : tensors[0].data()) {
        cout << v << " " << "\n";
    }
}