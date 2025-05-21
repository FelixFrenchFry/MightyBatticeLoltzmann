#pragma once
#include <vector>



void ComputeDensityField(
    const std::vector<float>& f,
    std::vector<float>& rho,
    const int N_CELLS);
