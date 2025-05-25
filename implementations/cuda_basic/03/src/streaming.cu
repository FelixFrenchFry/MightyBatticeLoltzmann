#include "config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



// load velocity direction vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_sk_c_x[9];
__constant__ int dvc_sk_c_y[9];
bool constantsInitialized_SK = false;

void InitializeConstants_SK()
{
    // one-time initialization guard
    if (constantsInitialized_SK) { return; }

    //  0: ( 0,  0) = rest
    //  1: ( 1,  0) = east
    //  2: ( 0,  1) = north
    //  3: (-1,  0) = west
    //  4: ( 0, -1) = south
    //  5: ( 1,  1) = north-east
    //  6: (-1,  1) = north-west
    //  7: (-1, -1) = south-west
    //  8: ( 1, -1) = south-east

    // initialize velocity direction vectors on the host
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_sk_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_sk_c_y, c_y, 9 * sizeof(int));

    cudaDeviceSynchronize();
    constantsInitialized_SK = true;
}

// TODO: only required to stream into 8 directions, because the center stays?
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeStreaming_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_next,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // declare and populate df_i in shared memory tile like df_tile[i][thread]
    __shared__ float df_tile[N_DIR][N_BLOCKSIZE];
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        df_tile[i][threadIdx.x] = dvc_df[i][idx];
    }
    // wait for data to be loaded
    __syncthreads();

    // determine coordinates of the source cell handled by this thread
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // determine coordinates and index within the SoA of the target cell
        // (with respect to periodic boundary conditions)
        uint32_t dst_x = (src_x + dvc_sk_c_x[i] + N_X) % N_X;
        uint32_t dst_y = (src_y + dvc_sk_c_y[i] + N_Y) % N_Y;
        uint32_t dst_idx = dst_y * N_X + dst_x;

        // stream distribution function value df_i to neighbor in direction i
        dvc_df_next[i][dst_idx] = df_tile[i][threadIdx.x];
    }
}

void Launch_StreamingComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    InitializeConstants_SK();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeStreaming_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_df_next, N_X, N_Y, N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA streaming kernel error: {}",
            cudaGetErrorString(err));
    }
}
