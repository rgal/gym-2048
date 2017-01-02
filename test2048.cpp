#include <stdio.h>
#include <assert.h>
#include "2048.h"

void compare_board(int a[SIZE * SIZE], int b[SIZE * SIZE]) {
    printf("Comparing board\n");
    for (int i = 0; i != SIZE * SIZE; i++) {
        if (a[i] != b[i]) {
            printf("Got: \n");
            print_board(a);
            printf("Expected: \n");
            print_board(b);
            assert(false);
            break;
        }
        //assert(a[i] == b[i]);
    }
}

void compare_array(int a[SIZE], int b[SIZE]) {
    printf("Comparing array\n");
    for (int i = 0; i != SIZE; i++) {
        if (a[i] != b[i]) {
            printf("Got: ");
            print_array(a);
            printf("Expected: ");
            print_array(b);
            assert(false);
            break;
        }
        //assert(a[i] == b[i]);
    }
}

void test_combine() {
    printf("test combine\n");
    int row[SIZE] = {2, 2, 4, 4};
    int expected[SIZE] = {4, 8, 0, 0};
    int score = combine(row);
    compare_array(row, expected);
    assert(score == 12);
}

void test_reverse() {
    printf("test reverse\n");
    int row[SIZE] = {1, 2, 4, 8};
    int expected[SIZE] = {8, 4, 2, 1};
    reverse(row);
    compare_array(row, expected);
}

void test_shift() {
    printf("test shift\n");
    int row[SIZE] = {0, 2, 2, 2};
    int expected[SIZE] = {4, 2, 0, 0};
    int score = shift(row, 0);
    compare_array(row, expected);
    assert(score == 4);
    int row2[SIZE] = {0, 2, 2, 8};
    int expected2[SIZE] = {0, 0, 4, 8};
    int score2 = shift(row2, 1);
    compare_array(row2, expected2);
    assert(score2 == 4);
}

void test_move() {
    printf("test move\n");
    // shift left
    int board[SIZE * SIZE] = {0, 2, 0, 4, 2, 2, 8, 0, 2, 2, 2, 8, 2, 2, 4, 4};
    print_board(board);
    int expected[SIZE * SIZE] = {2, 4, 0, 0, 4, 8, 0, 0, 4, 2, 8, 0, 4, 8, 0, 0};
    int score = move(board, 3);
    compare_board(board, expected);
    assert(score == 20);
    // shift up
    int board2[SIZE * SIZE] = {0, 2, 0, 4, 2, 2, 8, 0, 2, 2, 2, 8, 2, 2, 4, 4};
    print_board(board2);
    print_board(board2);
    print_board(board2);
    int expected2[SIZE * SIZE] = {4, 4, 8, 4, 2, 4, 2, 8, 0, 0, 4, 4, 0, 0, 0, 0};
    int score2 = move(board2, 0);
    compare_board(board2, expected2);
    assert(score2 == 12);
}

int main() {
    test_reverse();
    test_combine();
    test_shift();
    test_move();
    return 0;
}
