#include <array>
#include <vector>



void ComputeCollision(
    std::vector<float>& f,
    const std::vector<float>& rho,
    const std::vector<float>& u_x,
    const std::vector<float>& u_y,
    const std::array<float, 9>& w,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const float omega,
    const int N_CELLS)
{
    for (int i = 0; i < N_CELLS; i++)
    {
        // squared velocity
        float u_sq = u_x[i] * u_x[i] + u_y[i] * u_y[i];

        #pragma unroll
        for (int dir = 0; dir < 9; dir++)
        {
            // dot product of discrete direction c_i and velocity u
            float cu = c_x[dir] * u_x[i] + c_y[dir] * u_y[i];

            // equilibrium distribution function in direction i
            float f_eq_i = w[dir] * rho[i] * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

            // relax distribution function towards the equilibrium
            int idx = i * 9 + dir;
            f[idx] += omega * (f_eq_i - f[idx]);
        }
    }
}
