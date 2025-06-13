#include "../../tools/config.cuh"
#include "../../tools/utilities.h"
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
__constant__ FP dvc_fp_c_x[9];
__constant__ FP dvc_fp_c_y[9];
__constant__ FP dvc_w[9];
bool constantsInitialized = false;
bool kernelAttributesDisplayed = false;

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
    FP fp_c_x[9] = { 0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0 };
    FP fp_c_y[9] = { 0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0 };
    FP w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_opp_dir, opp_dir, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_fp_c_x, fp_c_x, 9 * sizeof(FP));
    cudaMemcpyToSymbol(dvc_fp_c_y, fp_c_y, 9 * sizeof(FP));
    cudaMemcpyToSymbol(dvc_w, w, 9 * sizeof(FP));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

// =============================================================================
// bounce-back boundary sub-kernels TODO: adjust for dealing with halo cells
// =============================================================================
__device__ __forceinline__ void ComputeNeighborIndex_BounceBackBoundary_Conditional_K(
    uint32_t src_x, uint32_t src_y, uint32_t src_y_global,
    uint32_t N_X, uint32_t N_Y_TOTAL,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // check if directed into a wall
    if ((dvc_c_x[i] == -1 && src_x == 0) ||               // into left wall
        (dvc_c_x[i] ==  1 && src_x == N_X - 1) ||         // into right wall
        (dvc_c_y[i] == -1 && src_y == 0) ||               // into bottom wall
        (dvc_c_y[i] ==  1 && src_y == N_Y_TOTAL - 1))     // into top wall
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
    uint32_t src_x, uint32_t src_y, uint32_t src_y_global,
    uint32_t N_X, uint32_t N_Y_TOTAL,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // TODO: reduce register usage
    // branch-less bit-wise computation of: 1 if bounce-back, else 0
    int bounce =
        ((dvc_c_x[i] == -1) & (src_x == 0)) |
        ((dvc_c_x[i] ==  1) & (src_x == N_X - 1)) |
        ((dvc_c_y[i] == -1) & (src_y == 0)) |
        ((dvc_c_y[i] ==  1) & (src_y == N_Y_TOTAL - 1));

    // branch-less computation of destination index
    dst_idx = bounce * (src_y * N_X + src_x)
            + (1 - bounce) * ((src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]));

    // branch-less computation of destination direction
    dst_i = bounce * dvc_opp_dir[i] + (1 - bounce) * i;
}

// =============================================================================
// inject lid velocity sub-kernels TODO: adjust for dealing with halo cells
// =============================================================================
__device__ __forceinline__ void InjectLidVelocity_Conditional_K(
    uint32_t src_y,
    uint32_t N_Y,
    FP rho,
    FP omega,
    FP u_lid,
    uint32_t i,
    FP& f_new_i)
{
    // check if directed into the top wall
    if (dvc_c_y[i] == 1 && src_y == N_Y - 1)
    {
        f_new_i -= FP_CONST(6.0) * omega * rho * dvc_fp_c_x[i] * u_lid;
    }
}

__device__ __forceinline__ void InjectLidVelocity_BranchLess_K(
    uint32_t src_y_global,
    uint32_t N_Y_TOTAL,
    FP rho,
    FP omega,
    FP u_lid,
    uint32_t i,
    FP& f_new_i)
{
    // branch-less lid velocity injection via boolean mask
    int top_bounce = ((dvc_c_y[i] == 1) & (src_y_global == N_Y_TOTAL - 1));

    f_new_i -= top_bounce * FP_CONST(6.0) * dvc_w[i] * rho
             * dvc_fp_c_x[i] * u_lid;
}

// =============================================================================
// fully fused lattice update kernel for lid shear wave decay simulation
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FullyFusedLatticeUpdate_ShearWaveDecay_Push_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* const* __restrict__ dvc_df_halo_top,
    FP* const* __restrict__ dvc_df_halo_bottom,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t Y_START, const uint32_t Y_END,
    const uint32_t N_CELLS,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // load df values into block-wise tiles of shared shared memory
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // used for summing stuff up and computing collision
    FP rho = FP_CONST(0.0);
    FP u_x = FP_CONST(0.0);
    FP u_y = FP_CONST(0.0);

    // populate shared memory tiles and compute sums in the same loop
    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        tile_df[i][threadIdx.x] = dvc_df[i][idx];
        rho += tile_df[i][threadIdx.x];
        u_x += tile_df[i][threadIdx.x] * dvc_fp_c_x[i];
        u_y += tile_df[i][threadIdx.x] * dvc_fp_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity and cell coordinates for this thread
    FP u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;
    uint32_t src_y_global = src_y + Y_START;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho
                  * (FP_CONST(1.0) + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega
                   * (tile_df[i][threadIdx.x] - f_eq_i);

        // TODO: inlined sub-kernel for the neighbor index
        // determine coordinates of the streaming destination cell
        // (with respect to periodic boundary conditions and halo cells)
        uint32_t dst_x = (src_x + dvc_c_x[i] + N_X) % N_X;
        uint32_t dst_y_raw = src_y + dvc_c_y[i]; // might not be within domain -> no %

        // check if streaming destination is outside of the process domain
        if (dst_y_raw < 0) // below -> stream into bottom halo
        {
            dvc_df_halo_bottom[i][dst_x] = f_new_i;
        }
        else if (dst_y_raw >= N_Y) // above -> stream into top halo
        {
            dvc_df_halo_top[i][dst_x] = f_new_i;
        }
        else // within -> stream to regular neighbor in regular df arrays
        {
            dvc_df_next[i][dst_y_raw * N_X + dst_x] = f_new_i;
        }
    }
}

// =============================================================================
// fully fused lattice update kernel for lid driven cavity simulation
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FullyFusedLatticeUpdate_LidDrivenCavity_Push_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* const* __restrict__ dvc_df_halo_top,
    FP* const* __restrict__ dvc_df_halo_bottom,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t Y_START, const uint32_t Y_END,
    const uint32_t N_CELLS,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // load df values into block-wise tiles of shared shared memory
    // TODO: add 1-layer halo cells?
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // used for summing stuff up and computing collision
    FP rho = FP_CONST(0.0);
    FP u_x = FP_CONST(0.0);
    FP u_y = FP_CONST(0.0);

    // populate shared memory tiles and compute sums in the same loop
    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        tile_df[i][threadIdx.x] = dvc_df[i][idx];
        rho += tile_df[i][threadIdx.x];
        u_x += tile_df[i][threadIdx.x] * dvc_fp_c_x[i];
        u_y += tile_df[i][threadIdx.x] * dvc_fp_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity and cell coordinates for this thread
    FP u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;
    uint32_t src_y_global = src_y + Y_START;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho
                  * (FP_CONST(1.0) + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega
                   * (tile_df[i][threadIdx.x] - f_eq_i);

        // determine coordinates and direction of the streaming destination cell
        // (with respect to bounce-back boundary conditions and halo cells)
        // check if streaming is directed into a wall (bounce-back)
        if ((dvc_c_x[i] == -1 && src_x == 0) ||                  // into left wall
            (dvc_c_x[i] ==  1 && src_x == N_X - 1) ||            // into right wall
            (dvc_c_y[i] == -1 && src_y_global == 0) ||           // into bottom wall
            (dvc_c_y[i] ==  1 && src_y_global == N_Y_TOTAL - 1)) // into top wall
        {
            // inject lid velocity if streaming is directed into top wall
            if (dvc_c_y[i] == 1 && src_y_global == N_Y_TOTAL - 1)
            {
                f_new_i -= FP_CONST(6.0) * omega * rho * dvc_fp_c_x[i] * u_lid;
            }

            // same cell but opposite direction because of bounce-back
            // (definitely within the process domain -> stream into regular df arrays)
            dvc_df_next[dvc_opp_dir[i]][src_y * N_X + src_x] = f_new_i;
        }
        else // (might be outside of the process domain)
        {
            uint32_t dst_x_raw = src_x + dvc_c_x[i];
            uint32_t dst_y_raw = src_y + dvc_c_y[i];

            // check if streaming destination is outside of the process domain
            if (dst_y_raw < 0) // below, but no wall -> stream into bottom halo
            {
                dvc_df_halo_bottom[i][dst_x_raw] = f_new_i;
            }
            else if (dst_y_raw >= N_Y) // above, but no wall -> stream into top halo
            {
                dvc_df_halo_top[i][dst_x_raw] = f_new_i;
            }
            else // within -> stream to regular neighbor in regular df arrays
            {
                dvc_df_next[i][dst_y_raw * N_X + dst_x_raw] = f_new_i;
            }
        }
    }
}

void Launch_FullyFusedLatticeUpdate_Push(
    const FP* const* dvc_df,
    FP* const* dvc_df_next,
    FP* const* dvc_df_halo_top,
    FP* const* dvc_df_halo_bottom,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t Y_START, const uint32_t Y_END,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS,
    const uint32_t N_PROCESSES,
    const int RANK,
    const bool shear_wave_decay,
    const bool lid_driven_cavity,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    // TODO: remove deprecated/unused kernel arguments
    if (shear_wave_decay)
    {
        FullyFusedLatticeUpdate_ShearWaveDecay_Push_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_df_halo_top, dvc_df_halo_bottom,
            dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y, N_X_TOTAL, N_Y_TOTAL,
            Y_START, Y_END, N_CELLS, write_rho, write_u_x, write_u_y);
    }
    else if (lid_driven_cavity)
    {
        FullyFusedLatticeUpdate_LidDrivenCavity_Push_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_df_halo_top, dvc_df_halo_bottom,
            dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y, N_X_TOTAL, N_Y_TOTAL,
            Y_START, Y_END, N_CELLS, write_rho, write_u_x, write_u_y);
    }
    else
    {
        if (RANK == 0) { SPDLOG_ERROR("No valid simulation scenario selected"); }
    }

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (RANK == 0 && !kernelAttributesDisplayed)
    {
        if (shear_wave_decay)
        {
            DisplayKernelAttributes(FullyFusedLatticeUpdate_ShearWaveDecay_Push_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FullyFusedLatticeUpdate_ShearWaveDecay_Push_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y, N_X_TOTAL, N_Y_TOTAL, N_STEPS, N_PROCESSES);
        }
        else if (lid_driven_cavity)
        {
            DisplayKernelAttributes(FullyFusedLatticeUpdate_LidDrivenCavity_Push_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FullyFusedLatticeUpdate_LidDrivenCavity_Push_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y, N_X_TOTAL, N_Y_TOTAL, N_STEPS, N_PROCESSES);
        }

        kernelAttributesDisplayed = true;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel of rank {} failed: {}",
            RANK, cudaGetErrorString(err));
    }
}
