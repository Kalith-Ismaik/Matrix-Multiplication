#include <cuda_runtime.h>                   // NVIDIA CUDA kernel support
#include <cuda_bf16.h>                      // NVIDIA half precision data support

#define TILE_SIZE 32
#define MANTISSA_BITS 16
#define WARP_SIZE 32

__global__ void BFPMatrixMultiplyKernel(const float* __restrict__ matrixA, const float* __restrict__ matrixB, float* __restrict__ matrixR,
                                        const float alpha, const float beta, const int SizeAX, const int SizeAY, const int SizeBX, const int SizeBY) {
    
    // Shared memory for input tiles
    __shared__ __nv_bfloat16 sharedA[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ __nv_bfloat16 sharedB[2][TILE_SIZE][TILE_SIZE + 1];
    
    // Local thread indices
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    
    // Calculate global thread indices
    const int row = blockIdx.x * TILE_SIZE + tid_x;
    const int col = blockIdx.y * TILE_SIZE + tid_y;
    
    // Register array for prefetching
    __nv_bfloat16 prefetchA[2] = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    __nv_bfloat16 prefetchB[2] = {__float2bfloat16(0.0f), __float2bfloat16(0.0f)};
    
    // Accumulator for floating point results
    __nv_bfloat16 final_sum = __float2bfloat16(0.0f);

    int currentBuffer = 0;

    const int numTiles = (SizeAY + TILE_SIZE - 1) / TILE_SIZE;

    // Prefetch first tile with bounds checking
    if (row < SizeAX && tid_y < SizeAY) {
        prefetchA[0] = __float2bfloat16(__ldg(&matrixA[row * SizeAY + tid_y]));
    }
    if (col < SizeBY && tid_x < SizeBX) {
        prefetchB[0] = __float2bfloat16(__ldg(&matrixB[col * SizeBX + tid_x]));
    }
    
    // Process tiles
    for (int tile = 0; tile < numTiles; ++tile) {

        int nextBuffer = 1 - currentBuffer;
        
        // Double buffering with bounds checking
        if (tile + 1 < numTiles) {
            const int nextTileOffset = (tile + 1) * TILE_SIZE;
            
            if (row < SizeAX && (nextTileOffset + tid_y) < SizeAY) {
                prefetchA[nextBuffer] = __float2bfloat16(__ldg(&matrixA[row * SizeAY + nextTileOffset + tid_y]));
            }
            
            if (col < SizeBY && (nextTileOffset + tid_x) < SizeBX) {
                prefetchB[nextBuffer] = __float2bfloat16(__ldg(&matrixB[col * SizeBX + nextTileOffset + tid_x]));
            }
        }

        // Store prefetched data to shared memory with explicit zero padding
        sharedA[currentBuffer][tid_x][tid_y] = (row < SizeAX && (tile * TILE_SIZE + tid_y) < SizeAY) ? prefetchA[currentBuffer] : __float2bfloat16(0.0f);
        sharedB[currentBuffer][tid_x][tid_y] = (col < SizeBY && (tile * TILE_SIZE + tid_x) < SizeBX) ? prefetchB[currentBuffer] : __float2bfloat16(0.0f);
        
        __syncthreads();

        // Process current tile in warp-sized chunks
        const int numWarpsInTile = (TILE_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        #pragma unroll
        for (int w = 0; w < numWarpsInTile; ++w) {

            const int warpStart = w * WARP_SIZE;
            const int remainingElements = min(WARP_SIZE, TILE_SIZE - warpStart);
            
            // Compute warp-level maximum values
            __nv_bfloat16 warp_maxA = __float2bfloat16(0.0f);
            __nv_bfloat16 warp_maxB = __float2bfloat16(0.0f);
            
            // Each thread finds its local max for current warp chunk
            __nv_bfloat16 local_maxA = __float2bfloat16(0.0f);
            __nv_bfloat16 local_maxB = __float2bfloat16(0.0f);
            
            #pragma unroll
            for (int k = 0; k < remainingElements; k++) {
                if (warpStart + k < TILE_SIZE) {
                    local_maxA = fmaxf(local_maxA, fabsf(sharedA[currentBuffer][tid_x][warpStart + k]));
                    local_maxB = fmaxf(local_maxB, fabsf(sharedB[currentBuffer][warpStart + k][tid_y]));
                }
            }
            
            // Warp-level reduction for max values
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                local_maxA = fmaxf(local_maxA, __shfl_down_sync(0xFFFFFFFF, local_maxA, offset));
                local_maxB = fmaxf(local_maxB, __shfl_down_sync(0xFFFFFFFF, local_maxB, offset));
            }
            
            // Broadcast max values to all threads in warp
            warp_maxA = __shfl_sync(0xFFFFFFFF, local_maxA, 0);
            warp_maxB = __shfl_sync(0xFFFFFFFF, local_maxB, 0);
            
            // Calculate scales
            const int max_mantissa = (1 << (MANTISSA_BITS - 1)) - 1;
            __nv_bfloat16 warp_scaleA = (warp_maxA > __float2bfloat16(1e-30f)) ? static_cast<__nv_bfloat16>(max_mantissa) / warp_maxA : __float2bfloat16(1.0f);
            __nv_bfloat16 warp_scaleB = (warp_maxB > __float2bfloat16(1e-30f)) ? static_cast<__nv_bfloat16>(max_mantissa) / warp_maxB : __float2bfloat16(1.0f);

            // Accumulate products with bounds checking
            __nv_bfloat16 warp_sum = __float2bfloat16(0.0f);
            __nv_bfloat16 warp_com = __float2bfloat16(0.0f);
            
            #pragma unroll
            for (int k = 0; k < remainingElements; k++) {
                if (warpStart + k < TILE_SIZE) {
                    __nv_bfloat16 valA = sharedA[currentBuffer][tid_x][warpStart + k];
                    __nv_bfloat16 valB = sharedB[currentBuffer][warpStart + k][tid_y];
                    
                    int quantA = __float2int_rn(valA * warp_scaleA);
                    int quantB = __float2int_rn(valB * warp_scaleB);
                    
                    quantA = max(-max_mantissa, min(quantA, max_mantissa));
                    quantB = max(-max_mantissa, min(quantB, max_mantissa));

                    __nv_bfloat16 prod = __hmul(__float2bfloat16(quantA), __float2bfloat16(quantB));
                    
                    __nv_bfloat16 y = __hsub(prod, warp_com);
                    __nv_bfloat16 t = __hadd(warp_sum, y);
                
                    warp_com = __hsub(__hsub(t, warp_sum), y);
                    warp_sum = t;

                    //warp_sum += __float2bfloat16(quantA * quantB);    //Uncomment this for half presion FMA arithmatics
                                                    //To implement this comment last 4 lines of codes of 
                                                    //Kahan summation algorithm.
                }
            }
            
            final_sum = __hadd(final_sum, __hdiv(warp_sum, __hmul(warp_scaleA, warp_scaleB)));
        }
        
        __syncthreads();
        currentBuffer = nextBuffer;
    }
    
    // Write final result with bounds checking
    if (row < SizeAX && col < SizeBY) {
        matrixR[row * SizeBY + col] = alpha * __bfloat162float(final_sum) + beta * matrixR[row * SizeBY + col];
    }
}

void matrixMultiplyLauncher(float* matrixA, float* matrixB, float* matrixR, float alpha, float beta, int SizeAX, int SizeAY, int SizeBX, int SizeBY) {
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((SizeAX + TILE_SIZE - 1) / TILE_SIZE,
                       (SizeBY + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaFuncSetCacheConfig(BFPMatrixMultiplyKernel, cudaFuncCachePreferL1);
    cudaFuncSetAttribute(BFPMatrixMultiplyKernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    
    BFPMatrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(matrixA, matrixB, matrixR, alpha, beta, SizeAX, SizeAY, SizeBX, SizeBY);
}
