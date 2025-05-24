#include <cuda_runtime.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include <vector>



void ExportScalarField(const float* dvc_buffer, int num_entries,
                       const std::string& fileName)
{
    std::vector<float> buffer(num_entries);

    cudaMemcpy(buffer.data(), &dvc_buffer, num_entries * sizeof(float),
        cudaMemcpyDeviceToHost);

    std::ofstream file(fileName, std::ios::binary);
    if (!file)
    {
        SPDLOG_ERROR("Could not open output file: {}", fileName);
        return;
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), num_entries * sizeof(float));
    file.close();
}
