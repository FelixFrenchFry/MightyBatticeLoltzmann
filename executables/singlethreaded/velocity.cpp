#include <array>
#include <vector>



void ComputeVelocityField(
    const std::vector<float>& f,
    const std::vector<float>& rho,
    std::vector<float>& u_x,
    std::vector<float>& u_y,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const int N_CELLS)
{
    for (int i = 0; i < N_CELLS; i++)
    {
        // guard against erroneous densities
        if (rho[i] <= 0.0f) [[unlikely]]
        {
            u_x[i] = 0.0f;
            u_y[i] = 0.0f;
            continue;
        }

        float sum_x = 0.0f;
        float sum_y = 0.0f;

        // sum up distribution functions, weighted for each direction
        #pragma unroll
        for (int dir = 0; dir < 9; dir++)
        {
            const float f_i = f[i * 9 + dir];
            sum_x += f_i * c_x[dir];
            sum_y += f_i * c_y[dir];
        }

        // divide by density for final velocity values
        u_x[i] = sum_x / rho[i];
        u_y[i] = sum_y / rho[i];
    }
}
