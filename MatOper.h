// MatOper.h
#ifndef MATOPER_H
#define MATOPER_H

#include <vector> 

float** ArrayAllocator(int x, int y);
void ArrayDeAllocator(float** array, int x);
void InitializeMatrixA(float** matrix, int x, int y);
void InitializeMatrixB(float** matrix, int x, int y);
void InitializeMatrixR(float** matrix, int x, int y);
void MatrixMultiplier2D(float** matrixA, float** matrixB, float** matrixR, int ax, int ay, int bx, int by);
void ErrCalculator(float** normal_result, float** bfp_result, int RX, int RY);
void StatCalculator(float* times, float& avg_time, float& std_dev, int NRuns);

#endif // MATOPER_H
