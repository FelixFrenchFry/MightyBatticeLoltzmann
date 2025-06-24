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

    // ---------
    // | 6 2 5 |
    // | 3 0 1 |
    // | 7 4 8 |
    // ---------

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
// fully fused lattice update kernel for shear wave decay sim
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_ShearWaveDecay_Push_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
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

    // pre-compute squared velocity and cell coordinates for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega * (tile_df[i][threadIdx.x] - f_eq_i);

        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        // determine destination cell's index based on x/y-coordinates and dir i
        // (with respect to periodic boundary conditions)
        // TODO: sign bug in (uint32_t + int) math?
        uint32_t dst_idx = ((src_y + dvc_c_y[i] + N_Y) % N_Y) * N_X
                         + ((src_x + dvc_c_x[i] + N_X) % N_X);

        // stream df value to the destination in dir i
        dvc_df_next[i][dst_idx] = f_new_i;
    }
}

// =============================================================================
// fully fused lattice update kernel for shear wave decay sim (branchless)
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_ShearWaveDecay_Push_BL_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
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

    // pre-compute squared velocity and cell coordinates for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega * (tile_df[i][threadIdx.x] - f_eq_i);

        // determine destination cell's index based on x/y-coordinates and dir i
        // (with respect to periodic boundary conditions)
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        // TODO: sign bug in (uint32_t + int) math?
        uint32_t dst_idx = ((src_y + dvc_c_y[i] + N_Y) % N_Y) * N_X
                         + ((src_x + dvc_c_x[i] + N_X) % N_X);

        // stream df value to the destination in dir i
        dvc_df_next[i][dst_idx] = f_new_i;
    }
}

// =============================================================================
// fully fused lattice update kernel for lid driven cavity sim
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_LidDrivenCavity_Push_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
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

    // pre-compute squared velocity and cell coordinates for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega * (tile_df[i][threadIdx.x] - f_eq_i);

        // determine coordinates and direction of the streaming destination cell
        // (with respect to bounce-back boundary conditions)
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        // check if streaming is directed into a wall (bounce-back)
        if ((dvc_c_x[i] == -1 && src_x == 0) ||        // into left wall
            (dvc_c_x[i] ==  1 && src_x == N_X - 1) ||  // into right wall
            (dvc_c_y[i] == -1 && src_y == 0) ||        // into bottom wall
            (dvc_c_y[i] ==  1 && src_y == N_Y - 1))    // into top wall
        {
            // inject lid velocity if streaming is directed into top wall
            if (dvc_c_y[i] == 1 && src_y == N_Y - 1)
            {
                // TODO: correct equation w.r.t. omega and dvc_w[i] ?
                f_new_i -= FP_CONST(6.0) * dvc_w[i] * rho * dvc_fp_c_x[i] * u_lid;
            }

            // same cell but opposite direction of dir i because of bounce-back
            dvc_df_next[dvc_opp_dir[i]][src_y * N_X + src_x] = f_new_i;
        }
        else // not directed into a wall
        {
            // stream df value to the destination in regular dir i
            dvc_df_next[i][(src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i])] = f_new_i;
        }
    }
}

// =============================================================================
// fully fused lattice update kernel for lid driven cavity sim (branchless)
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_LidDrivenCavity_Push_BL_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
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

    // pre-compute squared velocity and cell coordinates for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega * (tile_df[i][threadIdx.x] - f_eq_i);

        // determine coordinates and direction of the streaming destination cell
        // (with respect to bounce-back boundary conditions)
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        // branchless bit-wise computation of wall collision:
        // 1 if bounce-back, else 0
        uint32_t bounce =
            ((dvc_c_x[i] == -1) & (src_x == 0)) |        // into left wall
            ((dvc_c_x[i] ==  1) & (src_x == N_X - 1)) |  // into right wall
            ((dvc_c_y[i] == -1) & (src_y == 0)) |        // into bottom wall
            ((dvc_c_y[i] ==  1) & (src_y == N_Y - 1));   // into top wall

        // branchless computation of destination index
        uint32_t dst_idx = bounce * (src_y * N_X + src_x)
                         + (1 - bounce) * ((src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]));

        // branchless computation of destination direction
        uint32_t dst_i = bounce * dvc_opp_dir[i] + (1 - bounce) * i;

        // branchless lid velocity injection via boolean mask
        uint32_t top_bounce = ((dvc_c_y[i] == 1) & (src_y == N_Y - 1));
        f_new_i -= top_bounce
                 * FP_CONST(6.0) * dvc_w[i] * rho * dvc_fp_c_x[i] * u_lid;

        // stream df value to the destination in dir i
        // (dir i got reversed in case of bounce-back)
        dvc_df_next[dst_i][dst_idx] = f_new_i;
    }
}

void Launch_FullyFusedLatticeUpdate_Push(
    const FP* const* dvc_df,
    FP* const* dvc_df_next,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS,
    const bool shear_wave_decay,
    const bool lid_driven_cavity,
    const bool branchless,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    if (shear_wave_decay && !branchless)
    {
        FFLU_ShearWaveDecay_Push_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega,
            N_X, N_Y, N_CELLS, write_rho, write_u_x, write_u_y);
    }
    else if (shear_wave_decay && branchless)
    {
        FFLU_ShearWaveDecay_Push_BL_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega,
            N_X, N_Y, N_CELLS, write_rho, write_u_x, write_u_y);
    }
    else if (lid_driven_cavity && !branchless)
    {
        FFLU_LidDrivenCavity_Push_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid,
            N_X, N_Y, N_CELLS, write_rho, write_u_x, write_u_y);
    }
    else if (lid_driven_cavity && branchless)
    {
        FFLU_LidDrivenCavity_Push_BL_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid,
            N_X, N_Y, N_CELLS, write_rho, write_u_x, write_u_y);
    }
    else
    {
        SPDLOG_ERROR("No valid simulation scenario selected");
    }

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed)
    {
        if (shear_wave_decay && !branchless)
        {
            DisplayKernelAttributes(FFLU_ShearWaveDecay_Push_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_ShearWaveDecay_Push_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y);
        }
        else if (shear_wave_decay && branchless)
        {
            DisplayKernelAttributes(FFLU_ShearWaveDecay_Push_BL_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_ShearWaveDecay_Push_BL_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y);
        }
        else if (lid_driven_cavity && !branchless)
        {
            DisplayKernelAttributes(FFLU_LidDrivenCavity_Push_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_LidDrivenCavity_Push_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y);
        }
        else if (lid_driven_cavity && branchless)
        {
            DisplayKernelAttributes(FFLU_LidDrivenCavity_Push_BL_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_LidDrivenCavity_Push_BL_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y);
        }

        kernelAttributesDisplayed = true;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // specify detailed logging for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }
}
