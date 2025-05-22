// single-threaded CPU implementation of the Lattice-Boltzmann method

#include "collision.h"
#include "density.h"
#include "streaming.h"
#include "velocity.h"
#include "output/export.h"
#include <cmath>
#include <spdlog/spdlog.h>
#include <vector>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    // ----- INITIALIZATION OF PARAMETERS AND DATA STRUCTURES -----

    int N_X = 60;               // grid width
    int N_Y = 40;               // grid height
    int N_STEPS = 1;            // number of simulation steps
    int N_DIR = 9;              // number of velocity directions
    int N_CELLS = N_X * N_Y;    // number of grid cells

    float omega = 1.2f;         // relaxation factor
    float rho_0 = 1.0f;         // rest density
    float u_max = 0.1f;         // max velocity

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
    std::vector<float> rho(N_CELLS, rho_0);     // density field

    // collect buffer pointers in a struct for easy argument passing
    SimulationState state;
    state.f = &f;
    state.f_next = &f_next;
    state.u_x = &u_x;
    state.u_y = &u_y;
    state.rho = &rho;
    state.N_X = N_X;
    state.N_Y = N_Y;

    // ----- LBM SIMULATION LOOP -----

    for (size_t step = 1; step <= N_STEPS; step++)
    {
        ComputeDensityField(f, rho, N_CELLS);

        ComputeVelocityField(f, rho, u_x, u_y, c_x, c_y, N_CELLS);

        ComputeCollision(f, rho, u_x, u_y, w, c_x, c_y, omega, N_CELLS);

        ComputeStreaming(f, f_next, c_x, c_y, N_X, N_Y, N_CELLS);

        std::swap(f, f_next);

        ExportSimulationData(state, VelocityMagnitude, step, true);
    }

    return 0;
}
