#include <stdio.h>
#include <assert.h>

#define SIZE 4

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
