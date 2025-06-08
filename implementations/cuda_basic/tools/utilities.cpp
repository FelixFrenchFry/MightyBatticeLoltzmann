#include <cuda_runtime.h>
#include <spdlog/spdlog.h>



void DisplayDeviceModel()
{
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    SPDLOG_INFO("GPU: {} (CC: {})",
        props.name, props.major * 10 + props.minor);
}

void DisplayDeviceMemoryUsage()
{
    size_t bytes_free, bytes_total;
    cudaMemGetInfo(&bytes_free, &bytes_total);

    double mem_total = static_cast<double>(bytes_total) / (1 << 30);
    double mem_free = static_cast<double>(bytes_free)  / (1 << 30);
    double mem_used  = mem_total - mem_free;

    SPDLOG_INFO("Memory usage [total/used/free]: {:.3f} / {:.3f} / {:.3f} GB",
        mem_total, mem_used, mem_free);
    SPDLOG_INFO("------------------------------------------------------------");
}

void DisplayPerformanceStats(
    std::chrono::time_point<std::chrono::steady_clock> start_time,
    std::chrono::time_point<std::chrono::steady_clock> end_time,
    uint32_t N_X, uint32_t N_Y,
    uint32_t N_STEPS)
{
    double execution_time = std::chrono::duration<double>(end_time - start_time).count();

    uint64_t total_updates = static_cast<uint64_t>(N_X * N_Y)
                           * static_cast<uint64_t>(N_STEPS);

    double blups = static_cast<double>(total_updates) / (execution_time * 1e9);

    SPDLOG_INFO("----------------------------------------");
    SPDLOG_INFO("Total execution time:      {:.3f} sec", execution_time);
    SPDLOG_INFO("Step execution time:       {:.3f} ms", (execution_time / N_STEPS) * 1000.0f);
    SPDLOG_INFO("Simulation size [X/Y/N]:   [{}/{}/{}]", N_X, N_Y, N_STEPS);
    SPDLOG_INFO("BLUPS:                     {:.3f}", blups);
}
