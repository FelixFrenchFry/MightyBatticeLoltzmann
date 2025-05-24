#pragma once



void Initialize();

void Launch_InitializeDistributionFunction_K(float* dvc_distributionFunc,
                                             float initValue,
                                             int num_entries);

void Launch_ComputeDensityField_K(const float* dvc_distributionFunc,
                                  float* dvc_densityField,
                                  int num_cells);

void Launch_ComputeVelocityField_K(const float* dvc_distributionFunc,
                                   const float* dvc_densityField,
                                   float* dvc_velocityField_x,
                                   float* dvc_velocityField_y,
                                   int num_cells);

void Launch_StreamingStep_K(const float* dvc_distributionFunc,
                            float* dvc_distributionFunc_next,
                            int grid_width, int grid_height,
                            int num_cells);
