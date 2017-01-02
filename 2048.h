#ifndef _GUARD_2048_H
#define _GUARD_2048_H

#define SIZE 4

void print_board(int board[SIZE * SIZE]);
void print_array(int arr[SIZE]);
int combine(int row[SIZE]);
void reverse(int row[SIZE]);
int shift(int row[SIZE], int direction);
int move(int board[SIZE*SIZE], int direction);

#endif
