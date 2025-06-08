#pragma once
#include "config.cuh"
#include <cstdint>



FP* Launch_ComputeVelocityMagnitude_K(
    const FP* dvc_u_x,
    const FP* dvc_u_y,
    const uint32_t N_X,
    const uint32_t N_Y);
