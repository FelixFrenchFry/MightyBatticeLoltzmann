#include "config.cuh"
#include "data_export.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <sstream>
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

void ExportScalarFieldFromMPIDevices(
    const FP* dvc_buffer,
    const SimulationData type,
    MPI_Comm COMM,
    const std::string& outputDirName,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t Y_START,
    const uint32_t N_CELLS,
    const int RANK)
{
    std::vector<FP> host_buffer(N_CELLS);

    cudaError_t err = cudaMemcpy(host_buffer.data(), dvc_buffer,
        N_CELLS * sizeof(FP), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("Rank {} -> cudaMemcpy failed: {}",
                     RANK, cudaGetErrorString(err));
    }

    namespace fs = std::filesystem;
    fs::path outputPath = fs::path(outputDirName) / versionDirName / subDirName;

    if (RANK == 0)
    {
        try
        {
            SPDLOG_INFO("Rank 0 -> creating directory {}...", outputPath.string());

            fs::create_directories(outputPath);
        }
        catch (const fs::filesystem_error& e1)
        {
            SPDLOG_ERROR("Rank 0 -> failed to create directory: {}", e1.what());
        }
    }
    MPI_Barrier(COMM);

    std::string typeName = SimulationDataToString(type);
    std::ostringstream oss;
    oss << typeName << "_" << std::setw(9) << std::setfill('0') << suffixNum << ".bin";
    fs::path filename = outputPath / oss.str();

    // open shared file with MPI I/O
    SPDLOG_INFO("Rank {} -> opening shared file {}...", RANK, filename.string());
    // TODO: GETS STUCK HERE WHEN LAUNCHED WITH RANKS ON DIFFERENT GPUs
    MPI_File mpiFile;
    MPI_File_open(COMM, filename.string().c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &mpiFile);

    SPDLOG_INFO("Rank {} -> opened the file", RANK);

    // compute write-offset of this rank
    MPI_Offset byte_offset = static_cast<MPI_Offset>(Y_START) * N_X_TOTAL * sizeof(FP);

    // write this rank's N_CELLS data points to its section in the shared buffer
    MPI_File_write_at(mpiFile, byte_offset,
                  host_buffer.data(), N_CELLS,
                  FP_MPI_TYPE, MPI_STATUS_IGNORE);

    SPDLOG_INFO("Rank {} -> wrote to {} at offset y={} ({}x{})",
                RANK, filename.string(), Y_START, N_X, N_Y);

    MPI_File_close(&mpiFile);

    SPDLOG_INFO("Rank {} -> closed the file", RANK);
}

void ExportSimulationData(
    const SimulationExportContext& context,
    const SimulationData& type,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum)
{
    const FP* dvc_buffer = nullptr;
    FP* dvc_u_mag = nullptr;

    switch (type) {
    case VelocityMagnitude:
        // TODO: compute velocity magnitude and export it
        SPDLOG_ERROR("Export of velocity magnitude not implemented yet");
        return;

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
    ExportScalarFieldFromMPIDevices(dvc_buffer, type, context.COMM,
        context.outputDirName, versionDirName, subDirName, suffixNum,
        context.N_X, context.N_Y, context.N_X_TOTAL, context.N_Y_TOTAL,
        context.Y_START, context.N_CELLS, context.RANK);

    //if (dvc_u_mag != nullptr) { cudaFree(dvc_u_mag); }
}

void SelectWriteBackData(
    const uint32_t step,
    const uint32_t export_interval,
    bool export_rho,
    bool export_u_x,
    bool export_u_y,
    bool export_u_mag,
    bool& write_rho,
    bool& write_u_x,
    bool& write_u_y)
{
    if (step == 1 || step % export_interval == 0)
    {
        if (export_rho)                 { write_rho = true; }
        if (export_u_x || export_u_mag) { write_u_x = true; }
        if (export_u_y || export_u_mag) { write_u_y = true; }
    }
}

void ExportSelectedData(
    const SimulationExportContext context,
    const std::string export_name,
    const std::string export_num,
    const uint32_t step,
    const uint32_t export_interval,
    bool export_rho,
    bool export_u_x,
    bool export_u_y,
    bool export_u_mag)
{
    if (step == 1 || step % export_interval == 0)
    {
        if (export_rho)
        {
            ExportSimulationData(context,
                Density,
                export_num,
                export_name,
                step);
        }

        if (export_u_x)
        {
            ExportSimulationData(context,
                Velocity_X,
                export_num,
                export_name,
                step);
        }

        if (export_u_y)
        {
            ExportSimulationData(context,
                Velocity_Y,
                export_num,
                export_name,
                step);
        }

        if (export_u_mag)
        {
            ExportSimulationData(context,
                VelocityMagnitude,
                export_num,
                export_name,
                step);
        }
    }
}
