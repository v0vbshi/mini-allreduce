#include "../include/allreduce.hpp"
#include <cassert>

void allreduce(Tensor& t, Communicator& comm, int rank, int world_size) {
    assert(t.size() % world_size == 0);
    size_t chunk_size = t.size() / world_size;
    int right = (rank + 1) % world_size;
    int left = (rank - 1 + world_size) % world_size;
    // Phase 1: reduce scatter
    // After N-1 steps, each rank owns one fully reduced chunk
    for (int i = 0; i < world_size - 1; i++) {
        int send_chunk = (rank - i + world_size) % world_size;
        int recv_chunk = (rank - i - 1 + world_size) % world_size;

        // send chunk to the right neighbor
        comm.send(right, t.slice(send_chunk * chunk_size, chunk_size));

        // recieve chunk from the left neighbor
        Tensor incoming(chunk_size);
        comm.recv(left, incoming);

        // add recieved into local
        size_t offset = recv_chunk * chunk_size;
        for (size_t i = 0; i < chunk_size; i++) {
            t.data()[offset + i] += incoming.data()[i];
        }
    }

    // Phrase 2: All Gather 
    // After N-1 steps, each rank owns the full size data
    for (int i = 0; i < world_size - 1; i++) {
        int send_chunk = (rank + 1 -i + world_size) % world_size;
        int recv_chunk = (rank - i + world_size) % world_size;

        // send chunks
        comm.send(right, t.slice(send_chunk * chunk_size, chunk_size));

        // recieve chunks
        Tensor incoming(chunk_size);
        comm.recv(left, incoming);

        t.write_slice(recv_chunk * chunk_size, incoming);
    }
}