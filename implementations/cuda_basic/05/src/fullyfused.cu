#include "config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
__constant__ int dvc_c_x[9];
__constant__ int dvc_c_y[9];
__constant__ float dvc_w[9];
bool constantsInitialized = false;

void InitializeConstants()
{
    // one-time initialization guard
    if (constantsInitialized) { return; }

    //  0: ( 0,  0) = rest
    //  1: ( 1,  0) = east
    //  2: ( 0,  1) = north
    //  3: (-1,  0) = west
    //  4: ( 0, -1) = south
    //  5: ( 1,  1) = north-east
    //  6: (-1,  1) = north-west
    //  7: (-1, -1) = south-west
    //  8: ( 1, -1) = south-east

    // initialize velocity direction and weight vectors on the host
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    float w[9] = { 4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                   1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

// __restriced__ tells compiler there is no overlap among the data pointed to
// (reduces memory access and instructions, but increases register pressure!)
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeFullyFusedOperations_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_next,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float omega,
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
    // wait for data to be fully loaded
    __syncthreads();

    // ----- DENSITY COMPUTATION -----

    // used initially as sum and later as final velocities in computations
    float rho = 0.0f;

    // sum over distribution function values in each direction i
    // (SoA layout for coalesced memory access across threads)
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // load data from shared memory tile with local index
        rho += df_tile[i][threadIdx.x];
    }

    dvc_rho[idx] = rho;

    // ----- VELOCITY COMPUTATION -----

    // exit thread to avoid division by zero or erroneous values
    if (rho <= 0.0f)
    {
        dvc_u_x[idx] = 0.0f;
        dvc_u_y[idx] = 0.0f;
        return;
    }

    // used initially as sums and later as final velocities in computations
    float u_x = 0.0f;
    float u_y = 0.0f;

    // sum over distribution function values, weighted by each direction i
    // (SoA layout for coalesced memory access across threads)
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // load data from shared memory tile with local index
        float df_i = df_tile[i][threadIdx.x];
        u_x += df_i * dvc_c_x[i];
        u_y += df_i * dvc_c_y[i];
    }

    // divide sums by density to obtain final velocities
    u_x /= rho;
    u_y /= rho;
    dvc_u_x[idx] = u_x;
    dvc_u_y[idx] = u_y;

    // ----- COLLISION AND STREAMING COMPUTATION -----

    // load temp variables into read-only cache and multiple loads
    float u_sq = u_x * u_x + u_y * u_y;

    // determine coordinates of the source cell handled by this thread
    // TODO: bug in coordinate computation?
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    #pragma unroll 8 // limit unroll for lower register pressure
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // dot product of c_i * u (velocity directions times local velocity)
        float cu = static_cast<float>(dvc_c_x[i]) * u_x
                 + static_cast<float>(dvc_c_y[i]) * u_y;

        // compute equilibrium distribution f_eq_i for current direction i
        float f_eq_i = dvc_w[i] * rho
                     * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // relax distribution function towards equilibrium
        // TODO: bug in this optimized computation?
        float f_new_i = df_tile[i][threadIdx.x] * (1 - omega) + omega * f_eq_i;

        // determine coordinates and index within the SoA of the target cell
        // (with respect to periodic boundary conditions)
        uint32_t dst_idx = ((src_y + dvc_c_y[i] + N_Y) % N_Y) * N_X
                         + ((src_x + dvc_c_x[i] + N_X) % N_X);

        // stream distribution function value df_i to neighbor in direction i
        dvc_df_next[i][dst_idx] = f_new_i;
    }
}

void Launch_FullyFusedOperationsComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
        N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA fully fused kernel error: {}",
            cudaGetErrorString(err));
    }
}
