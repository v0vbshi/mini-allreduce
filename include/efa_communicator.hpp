#pragma once
#include "channel.hpp"
#include "communicator.hpp"
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>

class EfaCommunicator : public Communicator{
public:
    EfaCommunicator(int rank, int world_size, int sleep_ms,
        std::vector<std::shared_ptr<Channel>> channels,
        std::shared_ptr<std::atomic<int>> barrier_count,
        std::shared_ptr<std::mutex> barrier_mtx,
        std::shared_ptr<std::condition_variable> barrier_cv);

    void send(int to_rank, const Tensor& t) override;
    void recv(int from_rank, Tensor& t) override;
    void barrier() override;

private:
    int _rank;
    int _world_size;
    int _sleep_ms;
    std::vector<std::shared_ptr<Channel>> _channels;
    std::shared_ptr<std::atomic<int>> _barrier_count;
    std::shared_ptr<std::mutex> _barrier_mtx;
    std::shared_ptr<std::condition_variable> _barrier_cv;
};