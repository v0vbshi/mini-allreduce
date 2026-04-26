#include "../include/mock_communicator.hpp"
using namespace std;

MockCommunicator::MockCommunicator(int rank, int world_size, 
    vector<shared_ptr<Channel>> channels, shared_ptr<atomic<int>> barrier_count,
    shared_ptr<std::mutex> barrier_mtx,
    shared_ptr<std::condition_variable> barrier_cv)
    : _rank(rank), _world_size(world_size), _channels(channels),
    _barrier_count(barrier_count), _barrier_mtx(barrier_mtx), _barrier_cv(barrier_cv) {}

void MockCommunicator::send(int to_rank, const Tensor& t) {
    // Step 1: lock channel for each send
    {
        std::lock_guard<std::mutex> lock(_channels[_rank]->mtx);
        _channels[_rank]->queue.push(t.clone());
    }
    _channels[_rank]->cv.notify_one();
}

void MockCommunicator::recv(int from_rank, Tensor& t) {
    // Step 1: check lock for each recieve
    std::unique_lock<std::mutex> lock(_channels[from_rank]->mtx);
    _channels[from_rank]->cv.wait(lock, [&] {
        return !_channels[from_rank]->queue.empty();
    });
    t = std::move(_channels[from_rank]->queue.front());
    _channels[from_rank]->queue.pop();
}

void MockCommunicator::barrier() {
    std::unique_lock<std::mutex> lock(*_barrier_mtx);
    int gen = *_barrier_count / _world_size; // current generation
    (*_barrier_count)++;
    if (*_barrier_count % _world_size == 0) {
        _barrier_cv->notify_all();
    } else {
        // wake only when our generation completes, not a future barrier
        _barrier_cv->wait(lock, [&]{
            return (*_barrier_count / _world_size) > gen;
        });
    }
}