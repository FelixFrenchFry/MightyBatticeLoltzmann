#pragma once
#include <string>
#include <vector>



enum SimulationData
{
    VelocityMagnitude,
    Velocity_X,
    Velocity_Y,
    Density,
};

struct SimulationState
{
    const std::vector<float>* f = nullptr;
    const std::vector<float>* f_next = nullptr;
    const std::vector<float>* u_x = nullptr;
    const std::vector<float>* u_y = nullptr;
    const std::vector<float>* rho = nullptr;

    int N_X = 0;
    int N_Y = 0;
};

void ExportSimulationData(
    const SimulationState& state,
    const SimulationData type,
    const int step,
    const bool bin,
    const bool csv = false);

void ExportScalarField(
    const std::vector<float>& buffer,
    const std::string& fileName,
    const bool bin, const bool csv,
    const int N_X, const int N_Y);
