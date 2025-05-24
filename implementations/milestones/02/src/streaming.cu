#include <cuda_runtime.h>
#include <spdlog/spdlog.h>



// put velocity vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
__constant__ int dvc_vel_x[9];
__constant__ int dvc_vel_y[9];

void Initialize()
{
    //  0: ( 0,  0) = rest
    //  1: ( 1,  0) = east
    //  2: ( 0,  1) = north
    //  3: (-1,  0) = west
    //  4: ( 0, -1) = south
    //  5: ( 1,  1) = north-east
    //  6: (-1,  1) = north-west
    //  7: (-1, -1) = south-west
    //  8: ( 1, -1) = south-east

    // initialize velocity vectors on the host
    int vel_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int vel_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_vel_x, vel_x, sizeof(vel_x));
    cudaMemcpyToSymbol(dvc_vel_y, vel_y, sizeof(vel_y));

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA Error: {}", cudaGetErrorString(err));
    }
}

// ----- kernels -----

__global__ void InitializeDistributionFunction_K(float* dvc_distributionFunc,
                                                 float initValue,
                                                 int num_entries)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // guard to exit threads that fall outside the simulation domain
    // because threads are launched in groups (warps) of 32
    if (index >= num_entries) { return; }

    dvc_distributionFunc[index] = initValue;
}

__global__ void ComputeDensityField_K(const float* dvc_distributionFunc,
                                      float* dvc_densityField,
                                      int num_cells)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // guard to exit threads that fall outside the simulation domain
    // because threads are launched in groups (warps) of 32
    if (index >= num_cells) { return; }

    float densitySum = 0.0f;
    int base_index = index * 9;

    // sum over distribution function values in each direction
    // (unroll loop into 9 individual instructions, rather than a loop)
    #pragma unroll
    for (int dir = 0; dir < 9; dir++)
    {
        densitySum += dvc_distributionFunc[base_index + dir];
    }

    dvc_densityField[index] = densitySum;
}

__global__ void ComputeVelocityField_K(const float* dvc_distributionFunc,
                                       const float* dvc_densityField,
                                       float* dvc_velocityField_x,
                                       float* dvc_velocityField_y,
                                       int num_cells)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // guard to exit threads that fall outside the simulation domain
    // because threads are launched in groups (warps) of 32
    if (index >= num_cells) { return; }

    float local_density = dvc_densityField[index];

    // exit thread if the cell has no mass and therefore no velocity
    if (local_density <= 0.0f)
    {
        dvc_velocityField_x[index] = 0.0f;
        dvc_velocityField_y[index] = 0.0f;
        return;
    }

    float velocitySum_x = 0.0f;
    float velocitySum_y = 0.0f;
    int base_index = index * 9;

    // sum over distribution function values, weighted by each direction
    // (unroll loop into 9 individual instructions, rather than a loop)
    #pragma unroll
    for (int dir = 0; dir < 9; dir++)
    {
        velocitySum_x += dvc_distributionFunc[base_index + dir] * dvc_vel_x[dir];
        velocitySum_y += dvc_distributionFunc[base_index + dir] * dvc_vel_y[dir];
    }

    dvc_velocityField_x[index] = velocitySum_x / local_density;
    dvc_velocityField_y[index] = velocitySum_y / local_density;
}

// for each direction i, send distribution function component into that direction
// f_i(x, y) -> f_i(x + c_i.x, y + c_i.y)
__global__ void StreamingStep_K(const float* dvc_distributionFunc,
                                float* dvc_distributionFunc_next,
                                int grid_width, int grid_height,
                                int num_cells)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // guard to exit threads that fall outside the simulation domain
    // because threads are launched in groups (warps) of 32
    if (index >= num_cells) { return; }

    int cell_x = index % grid_width;
    int cell_y = index / grid_width;
    int base_index = index * 9;

    // "stream" (shift) distribution function components into each direction
    // (unroll loop into 9 individual instructions, rather than a loop)
    #pragma unroll
    for (int dir = 0; dir < 9; dir++)
    {
        // compute destiantion position, handling periodic boundary conditions
        int dest_x = (cell_x + dvc_vel_x[dir] + grid_width) % grid_width;
        int dest_y = (cell_y + dvc_vel_y[dir] + grid_height) % grid_height;
        int dest_index = (dest_y * grid_width + dest_x) * 9 + dir;

        // "stream" distributions to destination position
        dvc_distributionFunc_next[dest_index] = dvc_distributionFunc[base_index + dir];
    }
}

// ----- kernel launchers -----

void Launch_InitializeDistributionFunction_K(float* dvc_distributionFunc,
                                             float initValue,
                                             int num_entries)
{
    InitializeDistributionFunction_K<<<(num_entries + 255) / 256, 256>>>(
        dvc_distributionFunc, initValue, num_entries);

    // wait for device actions (kernels launch, memory copy, etc...) to finish
    // and report potential errors
    cudaDeviceSynchronize();

    SPDLOG_INFO("Initialized distribution function buffer.");

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel error: {}", cudaGetErrorString(err));
    }
}

void Launch_ComputeDensityField_K(const float* dvc_distributionFunc,
                                  float* dvc_densityField,
                                  int num_cells)
{
    ComputeDensityField_K<<<(num_cells + 255) / 256, 256>>>(
        dvc_distributionFunc, dvc_densityField, num_cells);

    // wait for device actions (kernels launch, memory copy, etc...) to finish
    // and report potential errors
    cudaDeviceSynchronize();

    SPDLOG_INFO("Computed density field.");

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel error: {}", cudaGetErrorString(err));
    }
}

void Launch_ComputeVelocityField_K(const float* dvc_distributionFunc,
                                   const float* dvc_densityField,
                                   float* dvc_velocityField_x,
                                   float* dvc_velocityField_y,
                                   int num_cells)
{
    ComputeVelocityField_K<<<(num_cells + 255) / 256, 256>>>(
        dvc_distributionFunc, dvc_densityField, dvc_velocityField_x,
        dvc_velocityField_y, num_cells);

    // wait for device actions (kernels launch, memory copy, etc...) to finish
    // and report potential errors
    cudaDeviceSynchronize();

    SPDLOG_INFO("Computed velocity field.");

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel error: {}", cudaGetErrorString(err));
    }
}

void Launch_StreamingStep_K(const float* dvc_distributionFunc,
                            float* dvc_distributionFunc_next,
                            int grid_width, int grid_height,
                            int num_cells)
{
    StreamingStep_K<<<(num_cells + 255) / 256, 256>>>(
        dvc_distributionFunc, dvc_distributionFunc_next, grid_width,
        grid_height, num_cells);

    // wait for device actions (kernels launch, memory copy, etc...) to finish
    // and report potential errors
    cudaDeviceSynchronize();

    SPDLOG_INFO("Completed streaming step.");

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel error: {}", cudaGetErrorString(err));
    }
}
