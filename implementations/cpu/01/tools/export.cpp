#include "export.h"
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>



void ExportSimulationData(
    const SimulationState& state,
    const SimulationData type,
    const int step,
    const bool bin,
    const bool csv)
{
    if (type == VelocityMagnitude)
    {
        // convert velocities into velocity magnitudes
        std::vector<float> u_mag(state.N_X * state.N_Y);

        for (int i = 0; i < state.N_X * state.N_Y; i++)
        {
            u_mag[i] = std::sqrt(state.u_x->at(i) * state.u_x->at(i)
                                 + state.u_y->at(i) * state.u_y->at(i));
        }

        ExportScalarField(u_mag,
            "velocity_magnitude_" + std::to_string(step),
            bin, csv, state.N_X, state.N_Y);
    }
    else if (type == Velocity_X)
    {
        ExportScalarField(*state.u_x,
            "velocity_x_" + std::to_string(step),
            bin, csv, state.N_X, state.N_Y);
    }
    else if (type == Velocity_Y)
    {
        ExportScalarField(*state.u_y,
            "velocity_y_" + std::to_string(step),
            bin, csv, state.N_X, state.N_Y);
    }
    else if (type == Density)
    {
        ExportScalarField(*state.rho,
            "density_" + std::to_string(step),
            bin, csv, state.N_X, state.N_Y);
    }
    else
    {
        SPDLOG_ERROR("Unknown simulation data type: " + std::to_string(type));
    }
}

void ExportScalarField(
    const std::vector<float>& buffer,
    const std::string& fileName,
    const bool bin, const bool csv,
    const int N_X, const int N_Y)
{
    std::streamsize size = buffer.size() * sizeof(float);

    // export data in .bin format
    if (bin)
    {
        std::ofstream file_bin(fileName + ".bin", std::ios::binary);
        if (!file_bin)
        {
            SPDLOG_ERROR("Could not open output file: {}.bin", fileName);
            return;
        }

        file_bin.write(reinterpret_cast<const char*>(buffer.data()), size);
        file_bin.close();

        SPDLOG_INFO("Exported data: {}",
            std::filesystem::current_path().string() + "/" + fileName + ".bin");
    }

    // export data in .csv format
    if (csv)
    {
        std::ofstream file_csv(fileName + ".csv");
        if (!file_csv)
        {
            SPDLOG_ERROR("Could not open output file: {}.csv", fileName);
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

        SPDLOG_INFO("Exported data: {}",
            std::filesystem::current_path().string() + "/" + fileName + ".csv");
    }
}
