// matrix_operations.cpp
#include <iostream>
#include <cmath>

#include "MatOper.h"

using namespace std;

// 2D DYNAMIC ARRAY ALLOCATOR
float** ArrayAllocator(int x, int y) {
    float** array = new float*[x];
    for (int i = 0; i < x; ++i) {
        array[i] = new float[y];
    }
    return array;
}

// 2D DYNAMIC ARRAY DEALLOCATOR
void ArrayDeAllocator(float** array, int x) {
    for (int i = 0; i < x; ++i) {
        delete[] array[i];
    }
    delete[] array;
}

// INITIALIZING THE INPUT ARRAY A
void InitializeMatrixA(float** matrix, int x, int y) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            matrix[i][j] = i/10000.0f - j/10000.0f;
        }
    }
}

// INITIALIZING THE INPUT ARRAY B
void InitializeMatrixB(float** matrix, int x, int y) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            matrix[i][j] = i/10000.0f + j/10000.0f;
        }
    }
}

// INITIALIZING THE INPUT ARRAY R
void InitializeMatrixR(float** matrix, int x, int y) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            matrix[i][j] = 0.0f;
        }
    }
}

// MATRIX-MATRIX MULTIPLICATION IN CPU
void MatrixMultiplier2D(float** matrixA, float** matrixB, float** matrixR, int ax, int ay, int bx, int by) {

    float alpha = 1.0f;
    float beta  = 0.0f;

    // CHECK THE VALIDITY OF MATRICES FOR MULTIPLICATION
    if (ay != bx) {
        cout << "Error: The number of columns in matrix A (" << ay
             << ") does not match the number of rows in matrix B (" << bx << ")." << endl;
        return;
    }

    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < by; ++j) {
            matrixR[i][j] = beta*(matrixR[i][j]);
            for (int k = 0; k < ay; ++k) {
                matrixR[i][j] += alpha*(matrixA[i][k] * matrixB[k][j]);
            }
        }
    }

    // Print the dimensions of the result matrix
    cout << "Matrix multiplication completed." << endl;
    cout << "The result matrix has dimensions: " << ax << " x " << by << endl;
}

// Function to calculate and print error metrics
void ErrCalculator(float** normal_result, float** bfp_result, int RX, int RY) {

    float sum_absolute_error = 0.0f;
    float sum_relative_error = 0.0f;

    int count = 0;

    for (int i = 0; i < RX; ++i) {
        for (int j = 0; j < RY; ++j) {
            
            float abs_error = abs(normal_result[i][j] - bfp_result[i][j]);
            sum_absolute_error += abs_error;

            if (abs(normal_result[i][j]) > 1e-30) {  // Avoid division by zero
                float relative_error = abs_error / abs(normal_result[i][j]);
                sum_relative_error += relative_error;
            }
            ++count;
        }
    }

    cout << "Mean absolute error: " << sum_absolute_error / count << endl;
    cout << "Mean relative error: " << sum_relative_error / count << endl;
}

// Calculate statistics
void StatCalculator(float* times, float& avg_time, float& std_dev, int NRuns) {

    for (int i = 0; i < NRuns; ++i) {
        avg_time += times[i];
    }
    avg_time /= NRuns;

    for (int i = 0; i < NRuns; ++i) {
        std_dev += pow(times[i] - avg_time, 2);
    }
    std_dev = sqrt(std_dev / NRuns);
}