#pragma once
#include <chrono>



// GPU model and CUDA compute capability version
void DisplayDeviceModel();

// GPU memory usage (Gibibytes -> 1 GB = 2^30 bytes = 1,073,741,824 bytes)
void DisplayDeviceMemoryUsage();

// execution time in seconds, number of lattice updates, blups
void DisplayPerformanceStats(
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::chrono::time_point<std::chrono::steady_clock> end_time,
    uint32_t N_X, uint32_t N_Y,
    uint32_t N_STEPS);
