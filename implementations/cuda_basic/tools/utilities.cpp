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
}
