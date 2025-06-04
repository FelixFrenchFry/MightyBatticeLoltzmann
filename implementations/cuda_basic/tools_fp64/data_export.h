#pragma once
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
    const double* const* dvc_df = nullptr;
    const double* const* dvc_df_next = nullptr;
    const double* dvc_rho = nullptr;
    const double* dvc_u_x = nullptr;
    const double* dvc_u_y = nullptr;
    double* dvc_temp = nullptr;
};

void ExportScalarFieldFromDevice(
    const double* dvc_buffer,
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
