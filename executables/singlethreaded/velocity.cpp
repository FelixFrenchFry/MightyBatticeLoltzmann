#include <array>
#include <vector>



void ComputeVelocityField(
    const std::vector<float>& f,
    const std::vector<float>& rho,
    std::vector<float>& v_x,
    std::vector<float>& v_y,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const int N_CELLS)
{
    for (int i = 0; i < N_CELLS; i++)
    {
        // guard against erroneous densities
        if (rho[i] <= 0.0f)
        {
            v_x[i] = 0.0f;
            v_y[i] = 0.0f;
            continue;
        }

        float sum_x = 0.0f;
        float sum_y = 0.0f;

        // sum up distribution functions, weighted for each direction
        #pragma unroll
        for (int k = 0; k < 9; k++)
        {
            const float f_i = f[i * 9 + k];
            sum_x += f_i * c_x[k];
            sum_y += f_i * c_y[k];
        }

        // divide by density for final velocity values
        v_x[i] = sum_x / rho[i];
        v_y[i] = sum_y / rho[i];
    }
}
