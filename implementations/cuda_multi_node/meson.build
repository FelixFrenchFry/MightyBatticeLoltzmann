# all exectuables for the multi-GPU, multi-node CUDA implementations using OpenMPI

executable(
    'cuda_multi_node_01',
    ['01/src/main.cpp'],
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [mpi, spdlog]
)
