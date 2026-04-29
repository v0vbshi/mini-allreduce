# mini-allreduce

Ring AllReduce in C++ — implements the collective communication algorithm used for gradient synchronization in distributed training (the same algorithm underlying NCCL and Trainium's Neuron Collectives), with an abstract communicator interface designed to swap between mock (threads) and real hardware (Neuron/EFA).

## What it does

Simulates N workers each holding a local tensor. After AllReduce, every worker holds the element-wise sum across all workers' tensors — exactly what happens during gradient synchronization in distributed training.

## The algorithm

Ring AllReduce runs in two phases:

**Phase 1: ReduceScatter** (N-1 steps)

Split the tensor into N chunks. At each step, each rank sends one chunk to its right neighbor and receives one chunk from its left neighbor, accumulating the sum. After N-1 steps, each rank owns the fully-reduced version of exactly one chunk.

```
Step 0:  rank0 --[chunk0]--> rank1 --[chunk1]--> rank2 --[chunk2]--> rank3 --[chunk3]--> rank0
         rank0 adds received chunk3
         rank1 adds received chunk0
         ...

After N-1 steps:
  rank0 owns chunk1 (fully summed across all ranks)
  rank1 owns chunk2
  rank2 owns chunk3
  rank3 owns chunk0
```

**Phase 2: AllGather** (N-1 steps)

Rotate the reduced chunks around the ring until every rank has all chunks.

```
After AllGather:
  rank0: [sum0, sum1, sum2, sum3]
  rank1: [sum0, sum1, sum2, sum3]
  rank2: [sum0, sum1, sum2, sum3]
  rank3: [sum0, sum1, sum2, sum3]
```

**Which chunk does each rank own after ReduceScatter?**

Counterintuitively, rank 0 receives chunk 3 first — but that chunk keeps moving. At each step rank 0 forwards its accumulated chunk to rank 1 and receives a new one. The chunk that *stops* at rank 0 is the last one it receives: chunk 1. The formula is `owned_chunk = (rank + 1) % N`.

**Bandwidth:** `2(N-1)/N * M` per rank — bandwidth-optimal regardless of rank count.

## Naive vs Ring AllReduce

**Naive AllReduce** (shared buffer, mutex-protected):
```
rank0 ──┐
rank1 ──┤──► [shared_buf] ──► broadcast to all ranks
rank2 ──┤         ▲
rank3 ──┘    (serialized — one rank writes at a time)

Bottleneck: single shared buffer. Bandwidth scales as O(1) regardless of N.
```

**Ring AllReduce** (P3 implementation):
```
Every rank talks only to its two neighbors simultaneously.
No shared buffer. All ranks active at every step.

    rank0 ──► rank1
      ▲              │
      │         rank2
      │              │
    rank3 ◄── rank2

All 4 links active simultaneously at every step.
Bandwidth per rank: 2(N-1)/N * M → approaches 2M as N grows.
```

For N=4, tensor size M: Ring uses **1.5x** the bandwidth of Naive and all ranks work in parallel. At N=128 (a trn1.32xlarge has 128 NeuronCores), Ring approaches 2M while Naive is still bottlenecked at one writer.

## Architecture

Three layers:

1. **Tensor** — move-only data container. No implicit copies.
2. **Communicator** — abstract interface (`send`, `recv`, `barrier`). Backend-agnostic.
3. **AllReduce** — the algorithm. Calls only `Communicator` methods, knows nothing about the transport.

Swap `MockCommunicator` (threads + queues) for a `NeuronCommunicator` (nccom API) and the algorithm runs unchanged on real Trainium hardware.

## Build and run

```bash
mkdir build && cd build
cmake .. && make
./mini-allreduce        # runs demo with 4 ranks
./test_allreduce        # runs GoogleTest suite
```

Requires: CMake 3.16+, C++17, internet connection (GoogleTest fetched automatically).

## Why move-only Tensor

`Tensor` disables implicit copies (`= delete`) and provides an explicit `.clone()` method. This prevents accidental GB-scale copies in hot paths — the same discipline PyTorch and NCCL enforce internally. Moves transfer ownership in O(1); copies require explicit intent.

## What a real Trainium port looks like

Swap `MockCommunicator` for a thin wrapper over Neuron Collectives (`nccom_allreduce`). The `allreduce()` function and `Tensor` class are unchanged — only the communicator implementation differs. This is the abstraction the interface was designed for.
