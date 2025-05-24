#include "minimal.cuh"
#include <iostream>



int main()
{
    std::cout << "Testing CUDA kernel reading from __constant__ memory...\n";

    cuda_test();

    std::cout << "B\n";

    launch_dummy_kernel();

    std::cout << "C\n";

    cuda_test();

    return 0;
}
