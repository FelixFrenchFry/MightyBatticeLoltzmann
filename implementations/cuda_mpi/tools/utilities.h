#pragma once
#include "config.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <spdlog/spdlog.h>
#include <unistd.h>



struct SimulationParameters
{
    // scale
    uint32_t N_X_TOTAL;
    uint32_t N_Y_TOTAL;
    uint32_t N_STEPS;

    // parameters
    FP omega;
    FP rho_0;
    FP u_max;
    FP u_lid;
    FP n_sin;

    // export
    uint32_t export_interval;
    char export_name[128];
    char export_num[128];
    bool export_rho;
    bool export_u_x;
    bool export_u_y;
    bool export_u_mag;

    // mode
    bool shear_wave_decay;
    bool lid_driven_cavity;

    // misc
    bool branchless_outer;
};

inline void OverwriteSimulationParameters(
    SimulationParameters& parameters,
    const std::string& key,
    const std::string& value)
{
    if (key == "N_X_TOTAL") parameters.N_X_TOTAL = std::stoi(value);
    else if (key == "N_Y_TOTAL") parameters.N_Y_TOTAL = std::stoi(value);
    else if (key == "N_STEPS") parameters.N_STEPS = std::stoi(value);
    else if (key == "omega") parameters.omega = std::stod(value);
    else if (key == "rho_0") parameters.rho_0 = std::stod(value);
    else if (key == "u_max") parameters.u_max = std::stod(value);
    else if (key == "u_lid") parameters.u_lid = std::stod(value);
    else if (key == "n_sin") parameters.n_sin = std::stod(value);
    else if (key == "export_interval") parameters.export_interval = std::stoi(value);
    else if (key == "export_name")
    {
        std::strncpy(parameters.export_name, value.c_str(), sizeof(parameters.export_name) - 1);
        parameters.export_name[sizeof(parameters.export_name) - 1] = '\0';
    }
    else if (key == "export_num")
    {
        std::strncpy(parameters.export_num, value.c_str(), sizeof(parameters.export_num) - 1);
        parameters.export_num[sizeof(parameters.export_num) - 1] = '\0';
    }
    else if (key == "export_rho") parameters.export_rho = (std::stoi(value) != 0);
    else if (key == "export_u_x") parameters.export_u_x = (std::stoi(value) != 0);
    else if (key == "export_u_y") parameters.export_u_y = (std::stoi(value) != 0);
    else if (key == "export_u_mag") parameters.export_u_mag = (std::stoi(value) != 0);
    else if (key == "shear_wave_decay") parameters.shear_wave_decay = (std::stoi(value) != 0);
    else if (key == "lid_driven_cavity") parameters.lid_driven_cavity = (std::stoi(value) != 0);
    else if (key == "branchless_outer") parameters.branchless_outer = (std::stoi(value) != 0);
    else
    {
        SPDLOG_WARN("Unknown parameter in simulation input file: {}", key);
    }
}

inline void DisplaySimulationParameters(
    SimulationParameters& parameters,
    int RANK = 0)
{
    if (RANK == 0)
    {
        printf("\nSimulation Parameters\n");
        printf("------------------------------\n");
        printf("%-20s = %u\n", "N_X_TOTAL", parameters.N_X_TOTAL);
        printf("%-20s = %u\n", "N_Y_TOTAL", parameters.N_Y_TOTAL);
        printf("%-20s = %u\n", "N_STEPS", parameters.N_STEPS);

        printf("%-20s = %.3f\n", "omega", parameters.omega);
        printf("%-20s = %.3f\n", "rho_0", parameters.rho_0);
        printf("%-20s = %.3f\n", "u_max", parameters.u_max);
        printf("%-20s = %.3f\n", "u_lid", parameters.u_lid);
        printf("%-20s = %.3f\n", "n_sin", parameters.n_sin);

        printf("%-20s = %u\n", "export_interval", parameters.export_interval);
        printf("%-20s = %s\n", "export_name", parameters.export_name);
        printf("%-20s = %s\n", "export_num", parameters.export_num);

        printf("%-20s = %s\n", "export_rho", parameters.export_rho ? "true" : "false");
        printf("%-20s = %s\n", "export_u_x", parameters.export_u_x ? "true" : "false");
        printf("%-20s = %s\n", "export_u_y", parameters.export_u_y ? "true" : "false");
        printf("%-20s = %s\n", "export_u_mag", parameters.export_u_mag ? "true" : "false");

        printf("%-20s = %s\n", "shear_wave_decay", parameters.shear_wave_decay ? "true" : "false");
        printf("%-20s = %s\n", "lid_driven_cavity", parameters.lid_driven_cavity ? "true" : "false");
        printf("\n");
    }
}


struct GPUInfo
{
    char name_host[256];
    char name_device[256];
    int RANK;
    int RANK_LOCAL;
    int cc;
    int sm_count;
    int clock_rate;
    int mem_clock_rate;

    size_t shared_mem_per_sm;
    size_t shared_mem_per_block;
    double mem_total;
    double mem_used;
    double mem_free;
};

inline GPUInfo GetDeviceInfos(
    int RANK,
    int RANK_LOCAL)
{
    GPUInfo res;

    // get gpu properties
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, RANK_LOCAL);

    // basic ranks
    res.RANK = RANK;
    res.RANK_LOCAL = RANK_LOCAL;

    // names
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    strncpy(res.name_host, hostname, 256);
    strncpy(res.name_device, props.name, 256);

    // pick GPU properties
    res.cc = props.major * 10 + props.minor;
    res.sm_count = props.multiProcessorCount;
    res.clock_rate = props.clockRate;
    res.mem_clock_rate = props.memoryClockRate;

    // shared memory capacities (bytes)
    res.shared_mem_per_sm = props.sharedMemPerMultiprocessor;
    res.shared_mem_per_block = props.sharedMemPerBlock;

    // global memory capacity/usage (Gibibytes -> 1 GB = 2^30 bytes = 1,073,741,824 bytes)
    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total);
    res.mem_total = static_cast<double>(bytes_total) / (1 << 30);
    res.mem_free = static_cast<double>(bytes_free)  / (1 << 30);
    res.mem_used  = res.mem_total - res.mem_free;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // specify detailed logging for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }

    return res;
}

inline void DisplayDeviceInfos(
    const std::vector<GPUInfo>& allInfo,
    uint32_t N_X, uint32_t N_Y,
    int RANK)
{
    if (RANK == 0)
    {
        printf("%-5s %-10s %-34s %-3s %-4s %-4s %-9s %-9s %-8s %-8s %-6s %-6s\n",
               "Rank", "Node", "GPU Model", "#", "CC", "SMs", "ShMem/SM",
               "Total GB", "Used GB", "Free GB", "Sub-X", "Sub-Y");
        printf("----------------------------------------------------------"
                     "----------------------------------------------------------\n");
        for (const auto& info : allInfo)
        {
            // limit node name to 8 characters + null
            char name_host_short[9];
            strncpy(name_host_short, info.name_host, 8);
            name_host_short[8] = '\0';

            printf("%-5d %-10s %-34s %-3d %-4d %-4d %-9lu %-9.3f %-8.3f %-8.3f %-6d %-6d\n",
                   info.RANK, name_host_short, info.name_device, info.RANK_LOCAL,
                   info.cc, info.sm_count, info.shared_mem_per_sm,
                   info.mem_total, info.mem_used, info.mem_free, N_X, N_Y);
        }
        printf("\n");
    }
}

// header-only display of CUDA kernel attributes
template <typename KernelT>
inline void DisplayKernelAttributes(
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
    SPDLOG_INFO("Halo cells per sub-domain: {:.2f} %\n",
            (2 * N_X * 100.0f) / (N_X * N_Y));
}

inline void DisplayProgressBar(
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

// execution time in seconds, number of lattice updates, blups
inline void DisplayPerformanceStats(
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
    SPDLOG_INFO("BLUPS:                     {:.3f}\n", blups);
}
