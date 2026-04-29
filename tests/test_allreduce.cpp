#include <gtest/gtest.h>
#include <thread>
#include "../include/test_utils.hpp"

TEST(AllReduceTest, TwoRanks) {
    std::vector<Tensor> tensors;
    tensors.reserve(2);
    for (int i = 0; i < 2; i++) {
        tensors.emplace_back(4);
        tensors[i].data() = {1.0f, 2.0f, 3.0f, 4.0f};
    }
    run_allreduce(tensors, 2);

    for (int r = 0; r < 2; r++) {
        EXPECT_FLOAT_EQ(tensors[r].data()[0], 2.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[1], 4.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[2], 6.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[3], 8.0f);
    }
}

TEST(AllReduceTest, FourRanks) {
    std::vector<Tensor> tensors;
    int ranks = 4;
    tensors.reserve(ranks);
    for (int i = 0; i < 4; i++) {
        tensors.emplace_back(4);
        tensors[i].data() = {1.0f, 2.0f, 3.0f, 4.0f};
    }
    run_allreduce(tensors, ranks);

    for (int r = 0; r < ranks; r++) {
        EXPECT_FLOAT_EQ(tensors[r].data()[0], 4.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[1], 8.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[2], 12.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[3], 16.0f);
    }
}

TEST(AllReduceTest, AllZeros) {
    std::vector<Tensor> tensors;
    int ranks = 2;
    tensors.reserve(ranks);
    for (int i = 0; i < ranks; i++) {
        tensors.emplace_back(4);
        tensors[i].data() = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    run_allreduce(tensors, ranks);

    for (int r = 0; r < ranks; r++) {
        EXPECT_FLOAT_EQ(tensors[r].data()[0], 0.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[1], 0.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[2], 0.0f);
        EXPECT_FLOAT_EQ(tensors[r].data()[3], 0.0f);
    }
}

