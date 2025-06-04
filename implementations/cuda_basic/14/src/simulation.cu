#include "config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load opposite direction vectors for bounce-back, velocity direction vectors,
// and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_opp_dir[9];
__constant__ int dvc_c_x[9];
__constant__ int dvc_c_y[9];
__constant__ double dvc_w[9];
bool constantsInitialized = false;

void InitializeConstants()
{
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

    // initialize opposite direction, velocity direction, and weight vectors
    int opp_dir[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    double w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_opp_dir, opp_dir, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_w, w, 9 * sizeof(double));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

template <uint32_t N_DIR>
__device__ __forceinline__ uint32_t ComputeDensityAndVelocity_K(
    const double* const* __restrict__ dvc_df,
    uint32_t idx,
    double& rho, double& u_x, double& u_y)
{
    rho = 0.0;
    u_x = 0.0;
    u_y = 0.0;

    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        double df_i = dvc_df[i][idx];
        rho += df_i;
        u_x += df_i * dvc_c_x[i];
        u_y += df_i * dvc_c_y[i];
    }
}

__device__ __forceinline__ void ComputeNeighborIndex_PeriodicBoundary_K(
    uint32_t src_x, uint32_t src_y,
    uint32_t N_X, uint32_t N_Y,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // determine index of the destination neihbor cell
    // (with respect to periodic boundary conditions)
    dst_idx = ((src_y + dvc_c_y[i] + N_Y) % N_Y) * N_X
            + ((src_x + dvc_c_x[i] + N_X) % N_X);
    dst_i = i;
}

__device__ __forceinline__ void ComputeNeighborIndex_BounceBackBoundary_Conditional_K(
    uint32_t src_x, uint32_t src_y,
    uint32_t N_X, uint32_t N_Y,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // check if directed into a wall
    if ((dvc_c_x[i] == -1 && src_x == 0) ||        // into left wall
        (dvc_c_x[i] ==  1 && src_x == N_X - 1) ||  // into right wall
        (dvc_c_y[i] == -1 && src_y == 0) ||        // into bottom wall
        (dvc_c_y[i] ==  1 && src_y == N_Y - 1))    // into top wall
    {
        // same cell but opposite direction because of bounce-back
        dst_idx = src_y * N_X + src_x;
        dst_i = dvc_opp_dir[i];
    }
    else
    {
        // normal neighbor in direction i
        dst_idx = (src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]);
        dst_i = i;
    }
}

__device__ __forceinline__ void ComputeNeighborIndex_BounceBackBoundary_BranchLess_K(
    uint32_t src_x, uint32_t src_y,
    uint32_t N_X, uint32_t N_Y,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // TODO: this increases register pressure by too much

    // branch-less bit-wise computation of: 1 if bounce-back, else 0
    int bounce =
        ((dvc_c_x[i] == -1) & (src_x == 0)) |
        ((dvc_c_x[i] ==  1) & (src_x == N_X - 1)) |
        ((dvc_c_y[i] == -1) & (src_y == 0)) |
        ((dvc_c_y[i] ==  1) & (src_y == N_Y - 1));

    // branch-less computation of destination index
    // TODO: reduce register usage
    /*
    uint32_t idx_normal = (src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]);
    uint32_t idx_bounce = src_y * N_X + src_x;
    dst_idx = bounce * idx_bounce + (1 - bounce) * idx_normal;
    */
    dst_idx = bounce * (src_y * N_X + src_x)
            + (1 - bounce) * ((src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]));

    // branch-less computation of destination direction
    dst_i = bounce * dvc_opp_dir[i] + (1 - bounce) * i;
}

__device__ __forceinline__ void InjectLidVelocity_Conditional_K(
    uint32_t src_y,
    uint32_t N_Y,
    double rho,
    double omega,
    double u_lid,
    uint32_t i,
    double& f_new_i)
{
    // check if directed into top wall
    if (dvc_c_y[i] == 1 && src_y == N_Y - 1)
    {
        f_new_i -= 6.0 * omega * rho * dvc_c_x[i] * u_lid;
    }
}

__device__ __forceinline__ void InjectLidVelocity_BranchLess_K(
    uint32_t src_y,
    uint32_t N_Y,
    double rho,
    double omega,
    double u_lid,
    uint32_t i,
    double& f_new_i)
{
    // branch-less lid velocity injection via boolean mask
    int top_bounce = ((dvc_c_y[i] ==  1) & (src_y == N_Y - 1));
    f_new_i -= top_bounce * 6.0 * dvc_w[i] * rho
             * static_cast<double>(dvc_c_x[i]) * u_lid;
}

template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeFullyFusedOperations_K(
    const double* const* __restrict__ dvc_df,
    double* const* __restrict__ dvc_df_next,
    double* __restrict__ dvc_rho,
    double* __restrict__ dvc_u_x,
    double* __restrict__ dvc_u_y,
    const double omega,
    const double u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // inlined sub-kernel for the density and velocity
    double rho, u_x, u_y;
    ComputeDensityAndVelocity_K<N_DIR>(
        dvc_df, idx, rho, u_x, u_y);

    // exit thread to avoid division by zero or erroneous values
    if (rho <= 0.0) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity and cell coordinates for this thread
    double u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        double cu = static_cast<double>(dvc_c_x[i]) * u_x
                  + static_cast<double>(dvc_c_y[i]) * u_y;
        double f_eq_i = dvc_w[i] * rho
                     * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);

        // relax df towards equilibrium
        double f_new_i = dvc_df[i][idx] - omega * (dvc_df[i][idx] - f_eq_i);

        // inlined sub-kernel for the neighbor index
        uint32_t dst_idx, dst_i;
        ComputeNeighborIndex_PeriodicBoundary_K(
            src_x, src_y, N_X, N_Y, i, dst_idx, dst_i);

        // inject lid velocity if directed into top wall
        //InjectLidVelocity_BranchLess_K(src_y, N_Y, rho, omega, u_lid, i,
        //    f_new_i);

        // stream df value df_i to the neighbor in dir i
        // (direction i gets reversed in case of bounce-back)
        dvc_df_next[dst_i][dst_idx] = f_new_i;
    }
}

void Launch_FullyFusedOperationsComputation(
    const double* const* dvc_df,
    double* const* dvc_df_next,
    double* dvc_rho,
    double* dvc_u_x,
    double* dvc_u_y,
    const double omega,
    const double u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y,
        N_CELLS, write_rho, write_u_x, write_u_y);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));
    }
}
