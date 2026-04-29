#include <iostream>
#include <chrono>
#include "../include/test_utils.hpp"

struct BenchResult {
    double gbps;
    double ms;
};


BenchResult bench(int world_size, size_t tensor_size) {
    std::vector<Tensor> tensors;
    tensors.reserve(world_size);
    for (int i = 0; i < world_size; i++) {
        tensors.emplace_back(tensor_size);
        for (float& v : tensors[i].data()) v = 1.0f;
    }
    auto start = std::chrono::high_resolution_clock::now();
    run_allreduce(tensors, world_size);
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double bytes = tensor_size * sizeof(float) * world_size;
    return {(bytes / seconds) / 1e9, seconds * 1000};
}

BenchResult bench_efa(int world_size, size_t tensor_size, int latency_us) {
    std::vector<Tensor> tensors;
    tensors.reserve(world_size);
    for (int i = 0; i < world_size; i++) {
        tensors.emplace_back(tensor_size);
        for (float& v : tensors[i].data()) v = 1.0f;
    }
    auto start = std::chrono::high_resolution_clock::now();
    run_efa_allreduce(tensors, world_size, latency_us);
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double bytes = tensor_size * sizeof(float) * world_size;
    return {(bytes / seconds) / 1e9, seconds * 1000};
}

int main() {
    std::cout << "=== Mock (in-process queues) ===\n";
    std::cout << "world_size | tensor_size | GB/s       | ms\n";
    std::cout << "-----------|-------------|------------|-------\n";
    for (int ws : {2, 4, 8}) {
        for (size_t ts : {65536, 1048576}) {
            auto r = bench(ws, ts);
            std::cout << ws << "          | " << ts << "       | " << r.gbps << " | " << r.ms << "\n";
        }
    }

    std::cout << "\n=== EFA mock (5us simulated latency per send) ===\n";
    std::cout << "world_size | tensor_size | GB/s       | ms\n";
    std::cout << "-----------|-------------|------------|-------\n";
    for (int ws : {2, 4, 8}) {
        for (size_t ts : {65536, 1048576}) {
            auto r = bench_efa(ws, ts, 5);
            std::cout << ws << "          | " << ts << "       | " << r.gbps << " | " << r.ms << "\n";
        }
    }
}
