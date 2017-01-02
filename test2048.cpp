#include "gtest/gtest.h"
#include "2048.h"

TEST(TestReverse, Simple) {
    int row[SIZE] = {1, 2, 4, 8};
    int expected[SIZE] = {8, 4, 2, 1};
    reverse(row);
    for (int i = 0; i != SIZE; i++) {
        EXPECT_EQ(expected[i], row[i]);
    }
}

TEST(TestCombine, Simple) {
    int row[SIZE] = {2, 2, 4, 4};
    int expected[SIZE] = {4, 8, 0, 0};
    int score = combine(row);
    int expected_score = 12;
    for (int i = 0; i != SIZE; i++) {
        EXPECT_EQ(expected[i], row[i]);
    }
    EXPECT_EQ(expected_score, score);
}

TEST(TestShift, Left) {
    int row[SIZE] = {0, 2, 2, 2};
    int expected[SIZE] = {4, 2, 0, 0};
    int score = shift(row, 0);
    int expected_score = 4;
    for (int i = 0; i != SIZE; i++) {
        EXPECT_EQ(expected[i], row[i]);
    }
    EXPECT_EQ(expected_score, score);
}

TEST(TestShift, Right) {
    int row[SIZE] = {0, 2, 2, 8};
    int expected[SIZE] = {0, 0, 4, 8};
    int score = shift(row, 1);
    int expected_score = 4;
    for (int i = 0; i != SIZE; i++) {
        EXPECT_EQ(expected[i], row[i]);
    }
    EXPECT_EQ(expected_score, score);
}

TEST(TestMove, Left) {
    int board[SIZE * SIZE] = {
        0, 2, 0, 4,
        2, 2, 8, 0,
        2, 2, 2, 8,
        2, 2, 4, 4
    };
    int expected[SIZE * SIZE] = {
        2, 4, 0, 0,
        4, 8, 0, 0,
        4, 2, 8, 0,
        4, 8, 0, 0
    };
    int score = move(board, 3);
    int expected_score = 20;
    for (int i = 0; i != SIZE * SIZE; i++) {
        EXPECT_EQ(expected[i], board[i]);
    }
    EXPECT_EQ(expected_score, score);
}

TEST(TestMove, Up) {
    int board[SIZE * SIZE] = {
        0, 2, 0, 4,
        2, 2, 8, 0,
        2, 2, 2, 8,
        2, 2, 4, 4
    };
    int expected[SIZE * SIZE] = {
        4, 4, 8, 4,
        2, 4, 2, 8,
        0, 0, 4, 4,
        0, 0, 0, 0
    };
    int score = move(board, 0);
    int expected_score = 12;
    for (int i = 0; i != SIZE * SIZE; i++) {
        EXPECT_EQ(expected[i], board[i]);
    }
    EXPECT_EQ(expected_score, score);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
