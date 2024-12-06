// GEMM.h
#ifndef GEMM_H
#define GEMM_H

void MatrixMultiplierLauncher(float** matrixA, float** matrixB, float** matrixR, int ax, int ay, int bx, int by);
void MatrixMultiplierCuBLAS(float** matrixA, float** matrixB, float** matrixR, int ax, int ay, int bx, int by);

#endif // GEMM_H