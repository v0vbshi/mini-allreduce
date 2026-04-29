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
    double ms = seconds * 1000;
    return {(bytes / seconds) / 1e9, ms};
}

int main() {
    std::cout << "world_size | tensor_size | GB/s\n";
    std::cout << "-----------|-------------|-----\n";
    for (int ws : {2, 4, 8, 16}) {
        for (size_t ts : {4, 1024, 65536, 1048576, 16777216}) {
            if (ts % ws != 0) continue; // skip invalid combos
            auto result = bench(ws, ts);
            std::cout << ws << "          | " << ts << "       | " << result.gbps << " GB/s | " << result.ms << " ms\n";
        }
    }
}
