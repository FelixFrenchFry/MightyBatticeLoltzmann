#include <array>
#include <cmath>
#include <vector>



void ApplyShearWaveCondition(
    std::vector<float>& f,
    std::vector<float>& u_x,
    std::vector<float>& u_y,
    std::vector<float>& rho,
    const std::array<float, 9>& w,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const float rho_0,
    const float u_max,
    const float k,
    const int N_X, const int N_Y)
{
    for (int y = 0; y < N_Y; y++)
    {
        // sinusoidal x-velocity for all cells at this y-coordinate
        const float u_x_val = u_max * std::sin(k * static_cast<float>(y));
        const float u_sq = u_x_val * u_x_val;

        for (int x = 0; x < N_X; x++)
        {
            const int idx = y * N_X + x;

            // initialize values of the different fields
            u_x[idx] = u_x_val;
            u_y[idx] = 0.0f;
            rho[idx] = rho_0;

            // initialize distribution functions as an equilibrium
            #pragma unroll
            for (int dir = 0; dir < 9; dir++)
            {
                // dot product of discrete direction c_i and velocity u
                const float cu = c_x[dir] * u_x_val + c_y[dir] * 0.0f;

                // equilibrium distribution function in direction i
                const float f_eq_i = w[dir] * rho_0
                    * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

                f[idx * 9 + dir] = f_eq_i;
            }
        }
    }
}
