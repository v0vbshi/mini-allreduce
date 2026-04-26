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

    Tensor clone() const {
        Tensor copy(_size);
        copy._data = _data;
        return copy;
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