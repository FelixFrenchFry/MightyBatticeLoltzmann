#pragma once
#include <array>
#include <vector>



void ComputeCollision(
    std::vector<float>& f,
    const std::vector<float>& rho,
    const std::vector<float>& u_x,
    const std::vector<float>& u_y,
    const std::array<float, 9>& w,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const float omega,
    const int N_CELLS);
