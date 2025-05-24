#include <cassert>
#include <vector>



void ComputeDensityField(
    const std::vector<float>& f,
    std::vector<float>& rho,
    const int N_CELLS)
{
    assert(f.size() == N_CELLS * 9);
    assert(rho.size() == N_CELLS);

    for (int i = 0; i < N_CELLS; i++)
    {
        float sum = 0.0f;

        // sum up distribution functions from each direction
        #pragma unroll
        for (int dir = 0; dir < 9; dir++)
        {
            sum += f[i * 9 + dir];
        }

        rho[i] = sum;
    }
}
