#pragma once
#include "tensor.hpp"

class Communicator {
public:
    // destructor
    virtual ~Communicator() = default;
    // pure virual methods
    virtual void send(int to_rank, const Tensor& t) = 0;
    virtual void recv(int from_rank, Tensor& t) = 0;
    virtual void barrier() = 0;

private:
};