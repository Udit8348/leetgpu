#include <iostream>

// __global__ indicates this function should be compiled for the device (not host)
// nvcc routes this function to to the cuda compiler and main to the host's compiler
__global__ void kernel( void ) {
}

int main( void ) {
    int blocksPerGrid
    kernel<<<1,1>>>();
    printf( "Hello, World!\n" );
    return 0;
}