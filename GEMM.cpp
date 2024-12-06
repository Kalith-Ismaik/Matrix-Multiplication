#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "GEMM.h"
#include "MatMul.h"

using namespace std;

// MATRIX-MATRIX MULTIPLICATION NAIVE GPU LAUNCHER
void MatrixMultiplierLauncher(float** matrixA, float** matrixB, float** matrixR, int ax, int ay, int bx, int by) {

    // Allocate host memory (flattened for CUDA compatibility)
    float* h_matrixA = new float[ax * ay];
    float* h_matrixB = new float[bx * by];
    float* h_result  = new float[ax * by];

    // Allocate device memory
    float* d_matrixA;
    float* d_matrixB;
    float* d_result;

    // CHECK THE VALIDITY OF MATRICES FOR MULTIPLICATION
    if (ay != bx) {
        cout << "Error: The number of columns in matrix A (" << ay
             << ") does not match the number of rows in matrix B (" << bx << ")." << endl;
        return;
    }

    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < ay; ++j) {
            h_matrixA[i * ay + j] = matrixA[i][j];
        }
    }

    for (int i = 0; i < by; ++i) {
        for (int j = 0; j < bx; ++j) {
            h_matrixB[i * bx + j] = matrixB[j][i];
        }
    }

    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < by; ++j) {
            h_result[i * by + j] = matrixR[i][j];
        }
    }

    // Use standard CUDA API to select the default device (device 0)
    int dev = 0;
    cudaError_t cudaStatus = cudaSetDevice(dev);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        return;
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed!" << std::endl;
        return;
    }

    // Allocate device memory
    if (cudaMalloc(&d_matrixA, ax * ay * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_matrixB, bx * by * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_result, ax * by * sizeof(float)) != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed!" << std::endl;
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_matrixA, h_matrixA, ax * ay * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, bx * by * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, ax * by * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Launch the kernel
    matrixMultiplyLauncher(d_matrixA, d_matrixB, d_result, alpha, beta, ax, ay, bx, by);

    // Check for kernel launch errors
    cudaError_t kernelStatus = cudaGetLastError();
    if (kernelStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelStatus) << std::endl;
        delete[] h_matrixA;
        delete[] h_matrixB;
        delete[] h_result;
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_result);
        cublasDestroy(handle);
        return;
    }
    
    // Copy the result from device to host
    cudaMemcpy(h_result, d_result, ax * by * sizeof(float), cudaMemcpyDeviceToHost);

    // Unflatten the result matrix
    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < by; ++j) {
            matrixR[i][j] = h_result[i * by + j];
        }
    }

    // Free device and host memory
    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_result;
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_result);
    cublasDestroy(handle);
}

// MATRIX-MATRIX MULTIPLICATION CuBLAS GPU LAUNCHER
void MatrixMultiplierCuBLAS(float** matrixA, float** matrixB, float** matrixR, int ax, int ay, int bx, int by) {

    // Allocate host memory (flattened for CUDA compatibility)
    float* h_matrixA = new float[ax * ay];
    float* h_matrixB = new float[bx * by];
    float* h_result  = new float[ax * by];

    // Allocate device memory
    float* d_matrixA;
    float* d_matrixB;
    float* d_result;

    // CHECK THE VALIDITY OF MATRICES FOR MULTIPLICATION
    if (ay != bx) {
        cout << "Error: The number of columns in matrix A (" << ay
             << ") does not match the number of rows in matrix B (" << bx << ")." << endl;
        return;
    }

    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < ay; ++j) {
            h_matrixA[i * ay + j] = matrixA[i][j];
        }
    }

    for (int i = 0; i < bx; ++i) {
        for (int j = 0; j < by; ++j) {
            h_matrixB[i * by + j] = matrixB[i][j];
        }
    }

    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < by; ++j) {
            h_result[i * by + j] = matrixR[i][j];
        }
    }

    // Use standard CUDA API to select the default device (device 0)
    int dev = 0;
    cudaError_t cudaStatus = cudaSetDevice(dev);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        return;
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed!" << std::endl;
        return;
    }

    // Allocate device memory
    if (cudaMalloc((void**)&d_matrixA, ax * ay * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_matrixB, bx * by * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_result, ax * by * sizeof(float)) != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed!" << std::endl;
        return;
    }

    // Copy data from host to device
    cudaMemcpy(d_matrixA, h_matrixA, ax * ay * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, bx * by * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, ax * by * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Launch the CUDA kernel
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                          by, ax, ay, 
                          &alpha, 
                          d_matrixB, CUDA_R_32F, by, 
                          d_matrixA, CUDA_R_32F, ay, 
                          &beta,
                          d_result, CUDA_R_32F, by, 
                          CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS GEMM execution failed!" << std::endl;
        delete[] h_matrixA;
        delete[] h_matrixB;
        delete[] h_result;
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_result);
        cublasDestroy(handle);
        return;
    }
    
    // Copy the result from device to host
    cudaMemcpy(h_result, d_result, ax * by * sizeof(float), cudaMemcpyDeviceToHost);

    // Unflatten the result matrix
    for (int i = 0; i < ax; ++i) {
        for (int j = 0; j < by; ++j) {
            matrixR[i][j] = h_result[i * by + j];
        }
    }

    // Free device and host memory
    delete[] h_matrixA;
    delete[] h_matrixB;
    delete[] h_result;
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_result);
    cublasDestroy(handle);
}
