#include <cuda_runtime.h>
#include <cstdio>



// define the __constant__ velocity vector
__constant__ int velocity[9];

__global__ void dummy_kernel()
{
    int tid = threadIdx.x;
    if (tid < 9)
    {
        printf("velocity[%d] = %d\n", tid, velocity[tid]);
    }
}

void launch_dummy_kernel()
{
    dummy_kernel<<<1, 9>>>();
}

void cuda_test()
{
    int host_vel[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };

    // copy from host to GPU's __constant__ memory
    cudaMemcpyToSymbol(velocity, host_vel, sizeof(host_vel));

    // launch a kernel that reads it
    launch_dummy_kernel();

    cudaDeviceSynchronize();
}
