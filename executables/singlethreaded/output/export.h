#pragma once
#include <string>



void ExportScalarField(
    const std::vector<float>& buffer,
    const std::string& fileName,
    const int N_X, const int N_Y);
