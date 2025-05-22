#pragma once

#include <string>



void ExportScalarField(const float* dvc_buffer, int num_entries,
                       const std::string& fileName);