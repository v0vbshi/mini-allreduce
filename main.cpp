#include <iostream>
#include "include/tensor.hpp"
#include "include/allreduce.hpp"
using namespace std;

// temp stub
class StubComm : public Communicator {
public:
    void send(int, const Tensor&) override {}
    void recv(int, Tensor&) override {}
    void barrier() override {}
};

int main() {
    Tensor t(4);
    cout << "size: " << t.size() << "\n";
    t.data() = {1.0f, 2.0f, 3.0f, 4.0f};

    StubComm comm;
    allreduce(t, comm, 0, 1);

    for (float v : t.data()) {
        cout << v << " ";
    }

    cout << "\n";
    return 0;
}