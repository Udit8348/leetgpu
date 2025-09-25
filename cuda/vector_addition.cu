#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // we only need to sum the N requested elements in parallel
    // threadsPerBlock is not constrained to problem size, so we need this conditional
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
        C[thread_id] = B[thread_id] + A[thread_id];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}