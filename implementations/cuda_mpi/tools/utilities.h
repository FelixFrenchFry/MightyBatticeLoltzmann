#pragma once
#include "config.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>
#include <unistd.h>



void inline DisplayProgressBar(
    const uint32_t step,
    const uint32_t N_STEPS)
{
    static int last_percent = -1;

    // remember last timestamp
    static auto last_time = std::chrono::steady_clock::now();

    float progress = static_cast<float>(step) / N_STEPS;
    int percent = static_cast<int>(progress * 100.0f);

    while (last_percent < percent)
    {
        last_percent++;

        // compute time diff for this percent step
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = now - last_time;
        last_time = now;

        double seconds_per_percent = diff.count();

        // estimate remaining time
        int remaining = 100 - last_percent;
        int eta_seconds = static_cast<int>(remaining * seconds_per_percent);

        int eta_h = eta_seconds / 3600;
        int eta_m = (eta_seconds % 3600) / 60;
        int eta_s = eta_seconds % 60;

        // temp simplified message info config
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] %v");

        // display colored progress milestone
        if (not (last_percent == 0 || last_percent == 100))
        {
            SPDLOG_INFO("\033[38;2;255;40;50m{:>3} %"
                        "\033[0m          (~ {:02}:{:02}:{:02} remaining, "
                        "step {}/{})",
                        last_percent, eta_h, eta_m, eta_s, step, N_STEPS);
        }
        else
        {
            SPDLOG_INFO("\033[38;2;255;40;50m{:>3} %\033[0m", last_percent);
        }

        // restore detailed message info config
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }

    if (step == N_STEPS) { std::cout << std::endl; }
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

    SPDLOG_INFO("Memory usage GB [total/used/free]: [ {:.3f} / {:.3f} / {:.3f} ]",
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

    std::cerr << std::endl;
    SPDLOG_INFO("Total execution time:      {:.3f} sec", execution_time);
    SPDLOG_INFO("Step execution time:       {:.3f} ms", (execution_time / N_STEPS) * 1000.0f);
    SPDLOG_INFO("BLUPS:                     {:.3f}", blups);
}

// header-only display of CUDA kernel attributes
template <typename KernelT>
void inline DisplayKernelAttributes(
    KernelT kernel,
    const std::string& kernel_name,
    uint32_t N_GRIDSIZE, uint32_t N_BLOCKSIZE,
    uint32_t N_X, uint32_t N_Y,
    uint32_t N_X_TOTAL, uint32_t N_Y_TOTAL,
    uint32_t N_STEPS,
    uint32_t N_PROCESSES)
{
    cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("Failed to get attributes for {}: {}",
            kernel_name, cudaGetErrorString(err));
        return;
    }

    SPDLOG_INFO("Kernel [name/grid/block]:  [ {} / {} / {} ]",
        kernel_name, N_GRIDSIZE, N_BLOCKSIZE);
    SPDLOG_INFO("Registers per thread:      {}", attr.numRegs);
    SPDLOG_INFO("Shared memory per block:   {} bytes", attr.sharedSizeBytes);
    SPDLOG_INFO("Local memory per thread:   {} bytes", attr.localSizeBytes);
    SPDLOG_INFO("Simulation size [X/Y/N]:   [ {} / {} / {} ]",
        N_X_TOTAL, N_Y_TOTAL, N_STEPS);
    SPDLOG_INFO("Sub-domain sizes [X/Y/N]:  [ {} / {} / {} ] x {}",
        N_X, N_Y, N_STEPS, N_PROCESSES);
    SPDLOG_INFO("Halo cells per sub-domain: {:.2f} %\n",
            (2 * N_X * 100.0f) / (N_X * N_Y));
}
