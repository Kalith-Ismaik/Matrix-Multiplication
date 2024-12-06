// C++ CUDA KERNEL IMPLEMENTATION OF 2D MATRIX-MATRIX MULTIPLICATION

#include <iostream>
#include <chrono>
#include <cmath>

#include "MatOper.h"
#include "GEMM.h"

using namespace std;
using namespace chrono;

int main() {

    int WRuns = 3;        // WARMUP RUNS TO STABILIZE GPU BEHAVIOUR AND CACHE HANDLING
    int NRuns = 10;       // BENCHMARK RUNS

    int SizeAX = 4096;    // ROWS IN MATRIX A
    int SizeAY = 4096;    // COLUMNS IN MATRIX A
    int SizeBX = 4096;    // ROWS IN MATRIX B
    int SizeBY = 16384;   // COLUMNS IN MATRIXB

    // ALLOCATE 2D ARRAYS FOR BOTH INPUT AND OUTPUT ARRAYS
    float** matrixA = ArrayAllocator(SizeAX, SizeAY);
    float** matrixB = ArrayAllocator(SizeBX, SizeBY);

    float** matrixR1 = ArrayAllocator(SizeAX, SizeBY);
    float** matrixR2 = ArrayAllocator(SizeAX, SizeBY);

    // INITIALIZE THE 2D ARRAY WITH DUMMY DATA
    InitializeMatrixA(matrixA, SizeAX, SizeAY);
    InitializeMatrixB(matrixB, SizeBX, SizeBY);

    for (int i = 0; i < WRuns; ++i) {

        InitializeMatrixR(matrixR1, SizeAX, SizeBY);
        InitializeMatrixR(matrixR2, SizeAX, SizeBY);

        // PERFORM MATRIX-MATRIX MULTIPLICATION IN NAIVE GPU KERNEL WITH BFP QUANTIZATION
        MatrixMultiplierLauncher(matrixA, matrixB, matrixR1, SizeAX, SizeAY, SizeBX, SizeBY);
        // PERFORM MATRIX MULTIPLICATION WITH CUBLAS LIBRARY
        MatrixMultiplierCuBLAS(matrixA, matrixB, matrixR2, SizeAX, SizeAY, SizeBX, SizeBY);

        cout << "Warmup run " << i + 1 << " completed" << endl;
    
    }

    float* CUBFP_times  = new float[NRuns];
    float* CUBLAS_times = new float[NRuns];

    for (int i = 0; i < NRuns; ++i) {

        InitializeMatrixR(matrixR1, SizeAX, SizeBY);
        InitializeMatrixR(matrixR2, SizeAX, SizeBY);

        // TIME THE START OF MATRIX MULTIPLICATION
        auto t1 = high_resolution_clock::now();

        // PERFORM MATRIX-MATRIX MULTIPLICATION IN NAIVE GPU KERNEL WITH BFP QUANTIZATION
        MatrixMultiplierLauncher(matrixA, matrixB, matrixR1, SizeAX, SizeAY, SizeBX, SizeBY);

        // TIME THE END OF MATRIX MULTIPLICATION
        auto t2 = high_resolution_clock::now();

        // CALCULATE THE TIME TAKEN FOR OPERATION
        CUBFP_times[i] = duration_cast<milliseconds>(t2 - t1).count();

        // TIME THE START OF MATRIX MULTIPLICATION
        auto t3 = high_resolution_clock::now();

        // PERFORM MATRIX MULTIPLICATION WITH CUBLAS LIBRARY
        MatrixMultiplierCuBLAS(matrixA, matrixB, matrixR2, SizeAX, SizeAY, SizeBX, SizeBY);

        // TIME THE END OF MATRIX MULTIPLICATION
        auto t4 = high_resolution_clock::now();

        // CALCULATE THE TIME TAKEN FOR OPERATION AND PRINT IT
        CUBLAS_times[i] = duration_cast<milliseconds>(t4 - t3).count();

        cout << "Benchmark run " << i + 1 << " completed" << endl;
    
    }

    // Print final statistics
    cout << "\nFinal Statistics over " << NRuns << " successful benchmark runs:" << endl;
    
    float cubfp_avg = 0.0f;
    float cubfp_std = 0.0f;
    StatCalculator(CUBFP_times, cubfp_avg, cubfp_std, NRuns);
    cout << "\nNaive BFP Matrix Multiplication Implementation with BF16 data handling:" << endl;
    cout << "  Average: " << cubfp_avg << " ms" << endl;
    cout << "  Std Dev: " << cubfp_std << " ms" << endl;

    float cublas_avg = 0.0f;
    float cublas_std = 0.0f;
    StatCalculator(CUBLAS_times, cublas_avg, cublas_std, NRuns);
    cout << "\nCublas Matrix Multiplication Implementation with Float32 data handling:" << endl;
    cout << "  Average: " << cublas_avg << " ms" << endl;
    cout << "  Std Dev: " << cublas_std << " ms" << endl;

    // CALCULATE THE PRECISION OF NAIVE VS CUBLAS MATRIX-MATRIX IMPLEMENTATION
    cout << "Error between Optimized Floating Point (cuBLAS GPU) matrix multiplication vs Naive Block Floating Point (GPU) matrix multiplication with 16 bit mantissa: " << endl;
    ErrCalculator(matrixR1, matrixR2, SizeAX, SizeBY);

    // DEALLOCATE THE 2D DYNAMIC ARRAY
    ArrayDeAllocator(matrixA, SizeAX);
    ArrayDeAllocator(matrixB, SizeBX);

    ArrayDeAllocator(matrixR1, SizeAX);
    ArrayDeAllocator(matrixR2, SizeAX);

    // DEALLOCATE THE 1D DYNAMIC ARRAYS
    delete[] CUBFP_times;
    delete[] CUBLAS_times;

    return 0;

}
