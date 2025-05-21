#include <array>
#include <vector>



void ComputeStreaming(
    const std::vector<float>& f,
    std::vector<float>& f_next,
    const std::array<int, 9>& c_x,
    const std::array<int, 9>& c_y,
    const int N_X, const int N_Y,
    const int N_CELLS)
{
    for (int y = 0; y < N_Y; y++)
    {
        for (int x = 0; x < N_X; x++)
        {
            // source cell of the "streams"
            int src_cell = y * N_X + x;

            #pragma unroll
            for (int dir = 0; dir < 9; dir++)
            {
                // compute destination position with periodic boundary conditions
                int x_dst = (x + c_x[dir] + N_X) % N_X;
                int y_dst = (y + c_y[dir] + N_Y) % N_Y;

                // destination cell of the stream in the direction
                int dst_cell = y_dst * N_X + x_dst;

                // distribution function value is streamed along the direction
                f_next[dst_cell * 9 + dir] = f[src_cell * 9 + dir];
            }
        }
    }
}
