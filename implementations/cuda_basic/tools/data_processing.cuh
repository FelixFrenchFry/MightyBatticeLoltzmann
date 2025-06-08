#pragma once
#include "config.cuh"
#include <cstdint>



float* Launch_ComputeVelocityMagnitude_K(
    const float* dvc_u_x,
    const float* dvc_u_y,
    const uint32_t N_X,
    const uint32_t N_Y);
