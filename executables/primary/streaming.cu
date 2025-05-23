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
template <int N_DIR> // specify loop count at compile time for optimizations
__global__ void ComputeStreaming_K_temp(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_next,
    const size_t N_X, const size_t N_Y,
    const size_t N_CELLS)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // coordinates of the source cell
    int src_x = idx % N_X;
    int src_y = idx / N_X;

    #pragma unroll
    for (int i = 0; i < N_DIR; i++)
    {
        // coordinates and index within the SoA of the target cell
        // (with respect to periodic boundary conditions)
        int dst_x = (src_x + dvc_sk_c_x[i] + N_X) % N_X;
        int dst_y = (src_y + dvc_sk_c_y[i] + N_Y) % N_Y;
        int dst_idx = dst_y * N_X + dst_x;

        // stream distribution function value df_i to neighbor in direction i
        dvc_df_next[i][dst_idx] = dvc_df[i][idx];
    }
}

void Launch_StreamingComputation_temp(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    const size_t N_X, const size_t N_Y,
    const size_t N_CELLS)
{
    InitializeConstants_SK();

    const int blockSize = 256;
    const int gridSize = (N_CELLS + blockSize - 1) / blockSize;

    ComputeStreaming_K_temp<9><<<gridSize, blockSize>>>(
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
