# all exectuables for the CUDA-aware MPI multi-GPU implementations

executable(
    'cuda_mpi_01',
    ['01/src/main.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog, mpi]
)
