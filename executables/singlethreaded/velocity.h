#pragma once
#include <array>
#include <vector>



void ComputeVelocityField(
    const std::vector<float>& f,
    const std::vector<float>& rho,
    std::vector<float>& v_x,
    std::vector<float>& v_y,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const int N_CELLS);
