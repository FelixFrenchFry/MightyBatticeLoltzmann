#include "export.h"
#include "helper.cuh"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <vector>



std::string SimulationDataToString(const SimulationData type)
{
    switch (type)
    {
    case VelocityMagnitude:     return "velocity_magnitude";
    case Velocity_X:            return "velocity_x";
    case Velocity_Y:            return "velocity_y";
    case Density:               return "density";
    default:                    return "unknown";
    }
}

std::string FormatStepSuffix(uint32_t step, uint32_t width = 6)
{
    std::ostringstream oss;
    oss << "_" << std::setw(width) << std::setfill('0') << step;
    return oss.str();
}

void ExportScalarFieldFromDevice(
    const float* dvc_buffer,
    const SimulationData type,
    const std::string& outputDirName,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum,
    const uint32_t N_X, const uint32_t N_Y,
    const bool bin, const bool csv)
{
    const uint32_t N_CELLS = N_X * N_Y;
    std::vector<float> buffer(N_CELLS);

    // ----- COPY DEVICE DATA TO HOST -----

    cudaError_t err = cudaMemcpy(buffer.data(), dvc_buffer,
        N_CELLS * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA memory copy failed in {}: {}",
            __func__, cudaGetErrorString(err));
        return;
    }

    // ----- HANDLE OUTPUT DIRECTORY AND FILE NAME -----

    namespace fs = std::filesystem;
    fs::path outputPath = fs::path(outputDirName) / versionDirName / subDirName;
    fs::create_directories(outputPath);

    std::string typeName = SimulationDataToString(type);
    std::string fileBase = typeName + FormatStepSuffix(suffixNum);
    fs::path filePathBase = outputPath / fileBase;

    std::streamsize size = buffer.size() * sizeof(float);

    // ----- EXPORT BINARY -----
    if (bin)
    {
        std::ofstream file_bin(filePathBase.string() + ".bin", std::ios::binary);
        if (!file_bin)
        {
            SPDLOG_ERROR("Could not open output file: {}.bin", filePathBase.string());
            return;
        }

        file_bin.write(reinterpret_cast<const char*>(buffer.data()), size);
        file_bin.close();

        SPDLOG_INFO("Exported data: (...)/{}", filePathBase.string() + ".csv");
    }

    // ----- EXPORT CSV -----
    if (csv)
    {
        std::ofstream file_csv(filePathBase.string() + ".csv");
        if (!file_csv)
        {
            SPDLOG_ERROR("Could not open output file: {}.csv", filePathBase.string());
            return;
        }

        for (int y = 0; y < N_Y; y++)
        {
            for (int x = 0; x < N_X; x++)
            {
                int idx = y * N_X + x;
                file_csv << buffer[idx];
                if (x < N_X - 1) { file_csv << ","; }
            }
            file_csv << "\n";
        }

        file_csv.close();

        SPDLOG_INFO("Exported data: (...)/{}", filePathBase.string() + ".csv");
    }
}

void ExportSimulationData(
    const SimulationExportContext& context,
    const SimulationData type,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum,
    const bool bin, const bool csv)
{
    const float* dvc_buffer = nullptr;
    float* dvc_u_mag = nullptr;

    switch (type)
    {
    case VelocityMagnitude:
        // compute velocity magnitude
        dvc_u_mag = Launch_ComputeVelocityMagnitude_K(
            context.dvc_u_x, context.dvc_u_y, context.N_X, context.N_Y);

        if (!dvc_u_mag)
        {
            SPDLOG_ERROR("Failed to compute velocity magnitude.");
            return;
        }

        dvc_buffer = dvc_u_mag;
        break;

    case Velocity_X:
        dvc_buffer = context.dvc_u_x;
        break;

    case Velocity_Y:
        dvc_buffer = context.dvc_u_y;
        break;

    case Density:
        dvc_buffer = context.dvc_rho;
        break;

    default:
        SPDLOG_ERROR("Unknown simulation data type passed to {}: {}",
                     __func__, static_cast<int>(type));
        return;
    }

    // launch the export using the export-type-dependent arguments
    ExportScalarFieldFromDevice(dvc_buffer, type, context.outputDirName,
        versionDirName, subDirName, suffixNum, context.N_X, context.N_Y, bin, csv);

    if (dvc_u_mag != nullptr) { cudaFree(dvc_u_mag); }
}
