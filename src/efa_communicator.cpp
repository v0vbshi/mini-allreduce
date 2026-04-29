#include "../include/efa_communicator.hpp"
#include <thread>
#include <stdexcept>
using namespace std;

// TODO: Real EFA port
// Replace send/recv bodies with libfabric calls:
//   fi_send(ep, buf, len, desc, dest_addr, context)  — post send buffer
//   fi_recv(ep, buf, len, desc, src_addr, context)   — post recv buffer
//   fi_cq_read(cq, entry, count)                     — poll completion queue
// Barrier maps to fi_barrier or nccom_barrier() from Neuron Collectives.

EfaCommunicator::EfaCommunicator(int rank, int world_size, int sleep_ms,
    vector<shared_ptr<Channel>> channels, shared_ptr<atomic<int>> barrier_count,
    shared_ptr<std::mutex> barrier_mtx,
    shared_ptr<std::condition_variable> barrier_cv,
    int timeout_ms)
    : _rank(rank), _world_size(world_size), _sleep_ms(sleep_ms), _timeout_ms(timeout_ms),
    _channels(channels), _barrier_count(barrier_count), _barrier_mtx(barrier_mtx),
    _barrier_cv(barrier_cv) {}

void EfaCommunicator::send(int to_rank, const Tensor& t) {
    if (to_rank < 0 || to_rank >= _world_size)
        throw std::out_of_range("send: to_rank " + std::to_string(to_rank) + " out of range");
    // Step 0: Sleep before send 
    std::this_thread::sleep_for(std::chrono::microseconds(_sleep_ms));
    // Step 1: lock channel for each send
    {
        std::lock_guard<std::mutex> lock(_channels[_rank]->mtx);
        _channels[_rank]->queue.push(t.clone());
    }
    _channels[_rank]->cv.notify_one();
}

void EfaCommunicator::recv(int from_rank, Tensor& t) {
    if (from_rank < 0 || from_rank >= _world_size)
        throw std::out_of_range("recv: from_rank " + std::to_string(from_rank) + " out of range");
    // Step 1: check lock for each receive
    std::unique_lock<std::mutex> lock(_channels[from_rank]->mtx);
    bool received = _channels[from_rank]->cv.wait_for(lock,
        std::chrono::milliseconds(_timeout_ms),
        [&]{ return !_channels[from_rank]->queue.empty(); });
    if (!received)
        throw std::runtime_error("recv timeout: rank " + std::to_string(_rank) +
            " waiting on rank " + std::to_string(from_rank));
    t = std::move(_channels[from_rank]->queue.front());
    _channels[from_rank]->queue.pop();
}

void EfaCommunicator::barrier() {
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