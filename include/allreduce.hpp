#pragma once
#include "tensor.hpp"
#include "communicator.hpp"

void allreduce(Tensor& t, Communicator& comm, int rank, int world_size);