#pragma once
#include <vector>

class Tensor {
public:
    // constructor
    Tensor(size_t n) : _data(n, 0.0f), _size(n) {}
    // move constructor
    Tensor(Tensor&& other) noexcept : _data(std::move(other._data)), _size(other._size) {other._size = 0;}
    // move assignment 
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            _data = std::move(other._data);
            _size = other._size;
            other._size = 0;
        }
        return *this;
    }

    // make a copy of the tensor
    Tensor clone() const {
        Tensor copy(_size);
        copy._data = _data;
        return copy;
    }

    // slice the tensor
    Tensor slice(size_t offset, size_t len) const {
        Tensor s(len);
        std::copy(_data.begin() + offset, _data.begin() + offset + len, s._data.begin());
        return s;
    }
    
    void write_slice(size_t offset, const Tensor& src) {
        std::copy(src._data.begin(), src._data.end(), _data.begin() + offset);
    }

    // deleted copy constructor
    Tensor(const Tensor&) = delete;
    // deleted copy assignment
    Tensor& operator = (const Tensor&) = delete;

    size_t size() const { return _size; }
    std::vector<float>& data() { return _data; }

private:
    std::vector<float> _data;
    size_t _size;
};