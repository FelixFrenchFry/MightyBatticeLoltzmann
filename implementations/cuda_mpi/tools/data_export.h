#pragma once
#include "config.cuh"
#include <string>



enum SimulationData
{
    VelocityMagnitude,
    Velocity_X,
    Velocity_Y,
    Density,
};

struct SimulationExportContext
{
    // pointers to simulation data on the device
    // (temp-array for temp storage of data derived from it)
    const FP* const* dvc_df = nullptr;
    const FP* const* dvc_df_next = nullptr;
    const FP* dvc_rho = nullptr;
    const FP* dvc_u_x = nullptr;
    const FP* dvc_u_y = nullptr;
    FP* dvc_temp = nullptr;

    // other infos
    std::string outputDirName = "exported";
    MPI_Comm COMM;
    uint32_t N_X = 0;
    uint32_t N_Y = 0;
    uint32_t N_X_TOTAL = 0;
    uint32_t N_Y_TOTAL = 0;
    uint32_t Y_START = 0;
    uint32_t N_CELLS = 0;
    int RANK = -1;
};

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
    const int RANK);

void ExportSimulationData(
    const SimulationExportContext& context,
    const SimulationData& type,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum);

void SelectWriteBackData(
    const uint32_t step,
    const uint32_t export_interval,
    bool export_rho,
    bool export_u_x,
    bool export_u_y,
    bool export_u_mag,
    bool& write_rho,
    bool& write_u_x,
    bool& write_u_y);

void ExportSelectedData(
    const SimulationExportContext context,
    const std::string export_name,
    const std::string export_num,
    const uint32_t step,
    const uint32_t export_interval,
    bool export_rho,
    bool export_u_x,
    bool export_u_y,
    bool export_u_mag);
