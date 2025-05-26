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
    const DF_Vec* __restrict__ dvc_df,
    DF_Vec* __restrict__ dvc_df_next,
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

    // vectorized load of first batch of 4 values into shared memory
    df_tile[0][threadIdx.x] = dvc_df[idx].df_0_to_3.x;
    df_tile[1][threadIdx.x] = dvc_df[idx].df_0_to_3.y;
    df_tile[2][threadIdx.x] = dvc_df[idx].df_0_to_3.z;
    df_tile[3][threadIdx.x] = dvc_df[idx].df_0_to_3.w;

    // vectorized load of second batch of 4 values into shared memory
    df_tile[4][threadIdx.x] = dvc_df[idx].df_4_to_7.x;
    df_tile[5][threadIdx.x] = dvc_df[idx].df_4_to_7.y;
    df_tile[6][threadIdx.x] = dvc_df[idx].df_4_to_7.z;
    df_tile[7][threadIdx.x] = dvc_df[idx].df_4_to_7.w;

    // default load of last value into shared memory
    df_tile[8][threadIdx.x] = dvc_df[idx].df_8;

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

    // declare struct for results (new df values to be streamed to neighbors)
    float df_new[9];

    #pragma unroll 8 // TODO: limit unroll for lower register pressure
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
        df_new[i] = df_tile[i][threadIdx.x] * (1 - omega) + omega * f_eq_i;
    }

    // store results into aligned struct for vectorized memory accesses
    DF_Vec result;
    result.df_0_to_3 = make_float4(df_new[0], df_new[1], df_new[2], df_new[3]);
    result.df_4_to_7 = make_float4(df_new[4], df_new[5], df_new[6], df_new[7]);
    result.df_8 = df_new[8];

    // determine coordinates and index within the SoA of the target cell
    // (with respect to periodic boundary conditions)
    uint32_t dst_x = (src_x + N_X) % N_X;
    uint32_t dst_y = (src_y + N_Y) % N_Y;
    uint32_t dst_idx = dst_y * N_X + dst_x;

    // stream distribution function values df_i to neighbor in directions i
    // (vectorized memory write to global memory)
    dvc_df_next[dst_idx] = result;
}

void Launch_FullyFusedOperationsComputation(
    const DF_Vec* dvc_df,
    DF_Vec* dvc_df_next,
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
        SPDLOG_ERROR("Kernel '{}' failed at line {}: {}",
                     __func__, __LINE__, cudaGetErrorString(err));
    }
}
