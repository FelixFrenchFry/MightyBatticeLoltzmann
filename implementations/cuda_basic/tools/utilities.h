#pragma once
#include "config.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>




void inline DisplayProgressBar(
    const uint32_t step,
    const uint32_t N_STEPS)
{
    float progress = static_cast<float>(step) / N_STEPS;
    uint32_t position = static_cast<int>(50 * progress);

    std::cout << "\r[";

    for (uint32_t i = 0; i < 50; i++)
    {
        if (i < position)       { std::cout << "="; }
        else if (i == position) { std::cout << ">"; }
        else                    { std::cout << " "; }
    }

    // RGB color for progress percentage
    std::cout << "] \033[38;2;255;40;50m"
              << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << " %"
              << "\033[0m"
              << " (step " << step << "/" << N_STEPS << ")";

    std::cout.flush();

    if (step == N_STEPS) { std::cout << "\n" << std::endl; }
}

// GPU model and CUDA compute capability version
void inline DisplayDeviceModel()
{
    std::cout << std::endl;

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    SPDLOG_INFO("GPU: {} (CC: {})", props.name, props.major * 10 + props.minor);
}

// GPU memory usage (Gibibytes -> 1 GB = 2^30 bytes = 1,073,741,824 bytes)
void inline DisplayDeviceMemoryUsage()
{
    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total);

    double mem_total = static_cast<double>(bytes_total) / (1 << 30);
    double mem_free = static_cast<double>(bytes_free)  / (1 << 30);
    double mem_used  = mem_total - mem_free;

    SPDLOG_INFO("Memory usage in GB [total/used/free]: [ {:.3f} / {:.3f} / {:.3f} ]",
        mem_total, mem_used, mem_free);
}

// execution time in seconds, number of lattice updates, blups
void inline DisplayPerformanceStats(
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::chrono::time_point<std::chrono::steady_clock> end_time,
    uint32_t N_X, uint32_t N_Y,
    uint32_t N_STEPS)
{
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    uint64_t total_updates = static_cast<uint64_t>(N_X * N_Y)
                           * static_cast<uint64_t>(N_STEPS);

    double blups = static_cast<double>(total_updates) / (execution_time * 1e9);

    SPDLOG_INFO("Total execution time:      {:.3f} sec", execution_time);
    SPDLOG_INFO("Step execution time:       {:.3f} ms", (execution_time / N_STEPS) * 1000.0f);
    SPDLOG_INFO("Simulation size [X/Y/N]:   [ {} / {} / {} ]", N_X, N_Y, N_STEPS);
    SPDLOG_INFO("BLUPS:                     {:.3f}", blups);
}

// header-only display of CUDA kernel attributes
template <typename KernelT>
void inline DisplayKernelAttributes(KernelT kernel, const std::string& kernel_name)
{
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("Failed to get attributes for {}: {}",
            kernel_name, cudaGetErrorString(err));
        return;
    }

    SPDLOG_INFO("Kernel attributes for:     {}", kernel_name);
    SPDLOG_INFO("Registers per thread:      {}", attr.numRegs);
    SPDLOG_INFO("Shared memory per block:   {} bytes", attr.sharedSizeBytes);
    SPDLOG_INFO("Local memory per thread:   {} bytes\n", attr.localSizeBytes);
}
