#include "streaming.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    SPDLOG_INFO("Starting up milestone02...");

    Streaming();

    return 0;
}
