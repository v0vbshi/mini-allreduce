#include "../include/allreduce.hpp"
#include <mutex>

static std::mutex mtx;
static std::vector<float> shared_buf;

void allreduce(Tensor& t, Communicator& comm, int rank, int world_size) {
    // Step 1: rank 0 initializes shared_buf
    if (rank == 0) {
        shared_buf.assign(t.size(), 0.0f);
    }
    // Step 2: barrier - wait for rank 0 to finish init
    comm.barrier();
    {
    // Step 3: add under lock - wait for all ranks
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t i = 0; i < t.size(); i++) {
            shared_buf[i] += t.data()[i];
        }
    }
    // Step 4: wo release locking with {} we could run into deadlock
    // barrier - wait for all ranks to finish adding 
    comm.barrier();

    // Step 5: every rank copies shared_buff back into t
    for (size_t i = 0; i < t.size(); i++) {
        t.data()[i] = shared_buf[i];
    }
}