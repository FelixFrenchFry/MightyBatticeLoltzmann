#include "config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// more GPU-efficient replacement for modulo operator
__device__ __forceinline__ int wrap(int val, int max)
{
    return (val < 0) ? val + max : (val >= max ? val - max : val);
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

    // ----- DENSITY COMPUTATION (MANUALLY UNROLLED) -----

    // used initially as sum and later as final velocities in computations
    float rho = 0.0f;

    // directions 0 to 8:
    rho += df_tile[0][threadIdx.x];
    rho += df_tile[1][threadIdx.x];
    rho += df_tile[2][threadIdx.x];
    rho += df_tile[3][threadIdx.x];
    rho += df_tile[4][threadIdx.x];
    rho += df_tile[5][threadIdx.x];
    rho += df_tile[6][threadIdx.x];
    rho += df_tile[7][threadIdx.x];
    rho += df_tile[8][threadIdx.x];

    dvc_rho[idx] = rho;

    // ----- VELOCITY COMPUTATION (MANUALLY UNROLLED) -----

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

    // direction 0: center
    // (x weight is 0 -> no contribution)
    // (y weight is 0 -> no contribution)

    // direction 1: east
    u_x += df_tile[1][threadIdx.x];
    // (y weight is 0 -> no contribution)

    // direction 2: north
    // (x weight is 0 -> no contribution)
    u_y += df_tile[2][threadIdx.x];

    // direction 3: west
    u_x -= df_tile[3][threadIdx.x];
    // (y weight is 0 -> no contribution)

    // direction 4: south
    // (x weight is 0 -> no contribution)
    u_y -= df_tile[4][threadIdx.x];

    // direction 5: north-east
    u_x += df_tile[5][threadIdx.x];
    u_y += df_tile[5][threadIdx.x];

    // direction 6: north-west
    u_x -= df_tile[6][threadIdx.x];
    u_y += df_tile[6][threadIdx.x];

    // direction 7: south-west
    u_x -= df_tile[7][threadIdx.x];
    u_y -= df_tile[7][threadIdx.x];

    // direction 8: south-east
    u_x += df_tile[8][threadIdx.x];
    u_y -= df_tile[8][threadIdx.x];

    // divide sums by density to obtain final velocities
    u_x /= rho;
    u_y /= rho;
    dvc_u_x[idx] = u_x;
    dvc_u_y[idx] = u_y;

    // ----- COLLISION AND STREAMING COMPUTATION (MANUALLY UNROLLED) -----

    // load temp variables into read-only cache and multiple loads
    float u_sq = u_x * u_x + u_y * u_y;

    // determine coordinates of the source cell handled by this thread
    // TODO: bug in coordinate computation?
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    constexpr float w1 = 1.0f/9.0f;
    constexpr float w2 = 1.0f/36.0f;

    // direction 0: center
    // (cu_0 = 0.0f)
    float f_eq_0 = (4.0f/9.0f) * rho * (1.0f - 1.5f * u_sq);
    float f_new_0 = df_tile[0][threadIdx.x] * (1 - omega) + omega * f_eq_0;
    uint32_t dst_idx_0 = src_y * N_X + src_x;
    dvc_df_next[0][dst_idx_0] = f_new_0;

    // direction 1: east
    float cu_1 = u_x;
    float f_eq_1 = w1 * rho * (1.0f + 3.0f * cu_1 + 4.5f * cu_1 * cu_1 - 1.5f * u_sq);
    float f_new_1 = df_tile[1][threadIdx.x] * (1 - omega) + omega * f_eq_1;
    uint32_t dst_idx_1 = src_y * N_X + wrap(src_x + 1, N_X);
    dvc_df_next[1][dst_idx_1] = f_new_1;

    // direction 2: north
    float cu_2 = u_y;
    float f_eq_2 = w1 * rho * (1.0f + 3.0f * cu_2 + 4.5f * cu_2 * cu_2 - 1.5f * u_sq);
    float f_new_2 = df_tile[2][threadIdx.x] * (1 - omega) + omega * f_eq_2;
    uint32_t dst_idx_2 = wrap(src_y + 1, N_Y) * N_X + src_x;
    dvc_df_next[2][dst_idx_2] = f_new_2;

    // direction 3: west
    float cu_3 = -u_x;
    float f_eq_3 = w1 * rho * (1.0f + 3.0f * cu_3 + 4.5f * cu_3 * cu_3 - 1.5f * u_sq);
    float f_new_3 = df_tile[3][threadIdx.x] * (1 - omega) + omega * f_eq_3;
    uint32_t dst_idx_3 = src_y * N_X + wrap(src_x - 1, N_X);
    dvc_df_next[3][dst_idx_3] = f_new_3;

    // direction 4: south
    float cu_4 = -u_y;
    float f_eq_4 = w1 * rho * (1.0f + 3.0f * cu_4 + 4.5f * cu_4 * cu_4 - 1.5f * u_sq);
    float f_new_4 = df_tile[4][threadIdx.x] * (1 - omega) + omega * f_eq_4;
    uint32_t dst_idx_4 = wrap(src_y - 1, N_Y) * N_X + src_x;
    dvc_df_next[4][dst_idx_4] = f_new_4;

    // direction 5: north-east
    float cu_5 = u_x + u_y;
    float f_eq_5 = w2 * rho * (1.0f + 3.0f * cu_5 + 4.5f * cu_5 * cu_5 - 1.5f * u_sq);
    float f_new_5 = df_tile[5][threadIdx.x] * (1 - omega) + omega * f_eq_5;
    uint32_t dst_idx_5 = wrap(src_y + 1, N_Y) * N_X + wrap(src_x + 1, N_X);
    dvc_df_next[5][dst_idx_5] = f_new_5;

    // direction 6: north-west
    float cu_6 = -u_x + u_y;
    float f_eq_6 = w2 * rho * (1.0f + 3.0f * cu_6 + 4.5f * cu_6 * cu_6 - 1.5f * u_sq);
    float f_new_6 = df_tile[6][threadIdx.x] * (1 - omega) + omega * f_eq_6;
    uint32_t dst_idx_6 = wrap(src_y + 1, N_Y) * N_X + wrap(src_x - 1, N_X);
    dvc_df_next[6][dst_idx_6] = f_new_6;

    // direction 7: south-west
    float cu_7 = -u_x - u_y;
    float f_eq_7 = w2 * rho * (1.0f + 3.0f * cu_7 + 4.5f * cu_7 * cu_7 - 1.5f * u_sq);
    float f_new_7 = df_tile[7][threadIdx.x] * (1 - omega) + omega * f_eq_7;
    uint32_t dst_idx_7 = wrap(src_y - 1, N_Y) * N_X + wrap(src_x - 1, N_X);
    dvc_df_next[7][dst_idx_7] = f_new_7;

    // direction 8: south-east
    float cu_8 = u_x - u_y;
    float f_eq_8 = w2 * rho * (1.0f + 3.0f * cu_8 + 4.5f * cu_8 * cu_8 - 1.5f * u_sq);
    float f_new_8 = df_tile[8][threadIdx.x] * (1 - omega) + omega * f_eq_8;
    uint32_t dst_idx_8 = wrap(src_y - 1, N_Y) * N_X + wrap(src_x + 1, N_X);
    dvc_df_next[8][dst_idx_8] = f_new_8;
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
