# all exectuables for the multi-GPU CUDA implementations

executable(
    'cuda_multi_01',
    ['01/src/main.cpp'],
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [mpi, spdlog]
)
