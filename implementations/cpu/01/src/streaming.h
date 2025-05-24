#pragma once
#include <array>
#include <vector>



void ComputeStreaming(
    const std::vector<float>& f,
    std::vector<float>& f_next,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const int N_X, const int N_Y,
    const int N_CELLS);
