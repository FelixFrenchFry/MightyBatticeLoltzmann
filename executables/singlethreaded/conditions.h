#pragma once
#include <array>
#include <vector>



void ApplyShearWaveCondition(
    std::vector<float>& f,
    std::vector<float>& u_x,
    std::vector<float>& u_y,
    std::vector<float>& rho,
    const std::array<float, 9>& w,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const float rho_0,
    const float u_max,
    const float k,
    const int N_X, const int N_Y);
