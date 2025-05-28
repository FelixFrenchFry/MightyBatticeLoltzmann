// single-threaded CPU implementation of the Lattice-Boltzmann method

#include "../src/streaming.h"
#include "../src/velocity.h"
#include "../tools/export.h"
#include "collision.h"
#include "conditions.h"
#include "density.h"
#include <spdlog/spdlog.h>
#include <vector>



int main(int argc, char* argv[])
{
    // ----- INITIALIZATION OF MISC STUFF -----

    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    constexpr float PI = 3.14159265f;

    // ----- INITIALIZATION OF PARAMETERS AND DATA STRUCTURES -----

    int N_X = 150;              // grid width
    int N_Y = 100;              // grid height
    int N_STEPS = 1000;         // number of simulation steps
    int N_DIR = 9;              // number of velocity directions
    int N_CELLS = N_X * N_Y;    // number of grid cells

    float omega = 1.5f;             // relaxation factor
    float rho_0 = 1.0f;             // rest density
    float u_max = 0.1f;             // max velocity
    float n = 1.0f;                 // number of full sine wave periods
    float k = (2.0f * PI * n)
        / static_cast<float>(N_Y);  // wavenumber (frequency)

    // weight vector, holding lattice weights for each velocity direction
    std::array<float, 9> w = {
        4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
        1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f };

    // velocity directions (x and y components separately)
    std::array<int, 9> c_x = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    std::array<int, 9> c_y = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    // initialize various buffers
    std::vector<float> f(N_CELLS * N_DIR);      // distribution function
    std::vector<float> f_next(N_CELLS * N_DIR); // updated distribution function
    std::vector<float> u_x(N_CELLS);            // velocity field (x components)
    std::vector<float> u_y(N_CELLS);            // velocity field (y components)
    std::vector<float> rho(N_CELLS);            // density field

    // apply initial conditions
    ApplyShearWaveCondition(f, u_x, u_y, rho, w, c_x,c_y,
                            rho_0, u_max, k, N_X, N_Y);

    // collect buffer references in a struct for easy argument passing
    SimulationState state;
    state.f = &f;
    state.f_next = &f_next;
    state.u_x = &u_x;
    state.u_y = &u_y;
    state.rho = &rho;
    state.N_X = N_X;
    state.N_Y = N_Y;

    // ----- LBM SIMULATION LOOP -----

    for (int step = 0; step <= N_STEPS; step++)
    {
        ComputeDensityField(f, rho, N_CELLS);

        ComputeVelocityField(f, rho, u_x, u_y, c_x, c_y, N_CELLS);

        ComputeCollision(f, rho, u_x, u_y, w, c_x, c_y, omega, N_CELLS);

        ComputeStreaming(f, f_next, c_x, c_y, N_X, N_Y, N_CELLS);

        std::swap(f, f_next);

        SPDLOG_INFO("--- step {} done ---", step);

        if (step % 100 == 0)
        {
            // evaluation of the simulation step
            ExportSimulationData(state, VelocityMagnitude, step, true);
        }
    }

    return 0;
}
