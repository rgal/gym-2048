#include <stdio.h>
#include <assert.h>
#include "2048.h"

extern "C" {
    int move(int board[SIZE*SIZE], int direction);
}

void print_board(int board[SIZE * SIZE]) {
    for (int y = 0; y != SIZE; y++) {
        for (int x = 0; x != SIZE; x++) {
            printf("%d ", board[y * SIZE + x]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_array(int arr[SIZE]) {
    for (int i = 0; i != SIZE; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int combine(int row[SIZE]) {
    // Combine by shifting to the left and combining identical numbers (pairwise)
    // Modifies row output
    int move_score = 0;
    int combined_row[SIZE];
    for (int i = 0; i != SIZE; i++) {
        combined_row[i] = 0;
    }
    bool skip = false;
    int output_index = 0;
    for (int i = 0; i != SIZE - 1; i++) {
        int this_one = row[i];
        int next_one = row[i+1];
        if (skip) {
            skip = false;
            continue;
        }
        combined_row[output_index] = this_one;
        if (this_one == next_one) {
            combined_row[output_index] += next_one;
            move_score += this_one + next_one;
            skip = true;
        }
        output_index++;
    }
    if (!skip) {
        combined_row[output_index] = row[SIZE-1];
    }

    for (int i = 0; i != SIZE; i++) {
        row[i] = combined_row[i];
    }
    // Returns score from that row
    return move_score;
}

void reverse(int row[SIZE]) {
    // Reverse input row
    for (int i = 0; i != (SIZE / 2); i++) {
        int temp = row[i];
        row[i] = row[SIZE - i - 1];
        row[SIZE - i - 1] = temp;
    }
}

int shift(int row[SIZE], int direction) {
    // Shift one row left (direction = 0) or right (direction = 1), combining if required.
    int score = 0;
    assert((direction == 0) || (direction == 1));
    int shifted_row[SIZE];
    int output_index = 0;
    //print_array(row);
    // Shift up (or down)
    for (int i = 0; i != SIZE; i++) {
        shifted_row[i] = 0;
    }
    if (direction)
        reverse(row);
    for (int i = 0; i != SIZE; i++) {
        if (row[i] != 0) {
            shifted_row[output_index] = row[i];
            output_index++;
        }
    }
    //print_array(shifted_row);

    score = combine(shifted_row);

    if (direction) {
        // Copy shifted_row to row and reverse
        for (int i = 0; i != SIZE; i++) {
            row[SIZE - i - 1] = shifted_row[i];
        }
    } else {
        for (int i = 0; i != SIZE; i++) {
            row[i] = shifted_row[i];
        }
    }

    return score;
}

int move(int board[SIZE*SIZE], int direction) {
    // Direction can be 0-3, equivalent to bearing in degrees divided by 90
    // board is the whole board in row major order
    // 0 up, 1 right, 2 down, 3 left
    // move direction     shift direction
    // 0 00                 0
    // 1 01                 1
    // 2 10                 1
    // 3 11                 0
    // 
    int score = 0;
    bool changed = false;
    int dir_mod_two = direction % 2;
    int dir_div_two = direction / 2;
    int shift_direction = dir_mod_two ^ dir_div_two;
 
    if (dir_mod_two == 0) {
        // Up or down, stride across 
        for (int x = 0; x != SIZE; x++) {
            int col[SIZE];
            for (int y = 0; y != SIZE; y++) {
                col[y] = board[y * SIZE + x];
            }
            score += shift(col, shift_direction);
            for (int y = 0; y != SIZE; y++) {
                if (col[y] != board[y * SIZE + x]) {
                    changed = true;
                }
                board[y * SIZE + x] = col[y];
            }
        }
    } else {
        // Left or right, choose bits
        for (int y = 0; y != SIZE; y++) {
            int row[SIZE];
            for (int x = 0; x != SIZE; x++) {
                row[x] = board[y * SIZE + x];
            }
            score += shift(row, shift_direction);
            for (int x = 0; x != SIZE; x++) {
                if (row[x] != board[y * SIZE + x]) {
                    changed = true;
                }
                board[y * SIZE + x] = row[x];
            }
        }
    }

    return score;
}

