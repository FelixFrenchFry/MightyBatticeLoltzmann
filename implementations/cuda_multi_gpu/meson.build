# all exectuables for the multi-GPU (single-node) CUDA implementations

executable(
    'cuda_multi_gpu_01',
    ['01/src/main.cpp'],
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [nccl, spdlog]
)
