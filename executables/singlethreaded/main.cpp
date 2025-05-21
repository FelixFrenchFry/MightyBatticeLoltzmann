#include "serial.h"
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    // grid size (number of lattice cells per dimension)
    int grid_width =    30;
    int grid_height =   20;

    // misc
    float relaxOmega = 1.2f;
    float restDensity = 1.0f;
    int num_cells = grid_width * grid_height;
    int num_dirs = 9;

    //  0: ( 0,  0) = rest
    //  1: ( 1,  0) = east
    //  2: ( 0,  1) = north
    //  3: (-1,  0) = west
    //  4: ( 0, -1) = south
    //  5: ( 1,  1) = north-east
    //  6: (-1,  1) = north-west
    //  7: (-1, -1) = south-west
    //  8: ( 1, -1) = south-east

    // initialize weight and velocity vectors
    float weights[9] = { 4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                         1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f };
    int vel_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int vel_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    // initialize distribution function vectors
    std::vector<float> dist(num_cells * num_dirs, 1.0f);
    std::vector<float> dist_next(num_cells * num_dirs, 1.0f);
}
