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
    std::string outputDirName = "exported";
    uint32_t N_X = 0;
    uint32_t N_Y = 0;

    // pointers to simulation data on the device
    // (temp-array for temp storage of data derived from it)
    const FP* const* dvc_df = nullptr;
    const FP* const* dvc_df_next = nullptr;
    const FP* dvc_rho = nullptr;
    const FP* dvc_u_x = nullptr;
    const FP* dvc_u_y = nullptr;
    FP* dvc_temp = nullptr;
};

void ExportScalarFieldFromDevice(
    const FP* dvc_buffer,
    const SimulationData type,
    const std::string& outputDirName,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum,
    const uint32_t N_X, const uint32_t N_Y,
    const bool bin, const bool csv);

void ExportSimulationData(
    const SimulationExportContext& context,
    const SimulationData type,
    const std::string& versionDirName,
    const std::string& subDirName,
    const uint32_t suffixNum,
    const bool bin = true, const bool csv = false);

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
