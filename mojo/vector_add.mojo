from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn vector_add_kernel(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    # https://docs.modular.com/mojo/manual/types/#numeric-type-conversion cannot compare SIMD with Int32 without an explicit ctor call
    if Int32(tid) < N:
        C[tid] = A[tid] + B[tid]

# A, B, C are device pointers (i.e. pointers to memory on the GPU)
@export                         
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var BLOCK_SIZE: Int32 = 256
    var ctx = DeviceContext()
    var num_blocks = ceildiv(N, BLOCK_SIZE)

    ctx.enqueue_function[vector_add_kernel](
        A, B, C, N,
        grid_dim  = num_blocks,
        block_dim = BLOCK_SIZE
    )

    ctx.synchronize()