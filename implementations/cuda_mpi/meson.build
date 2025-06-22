# all exectuables for the CUDA-aware MPI multi-GPU implementations

executable(
    'cuda_mpi_01',
    ['01/src/main.cpp',
     '01/src/initialization.cu',
     '01/src/simulation.cu',
     'tools/data_export.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog, cuda, mpi]
)

executable(
    'cuda_mpi_02',
    ['02/src/main.cpp',
     '02/src/initialization.cu',
     '02/src/simulation.cu',
     'tools/data_export.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog, cuda, mpi]
)

executable(
    'cuda_mpi_03',
    ['03/src/main.cpp',
     '03/src/initialization.cu',
     '03/src/simulation.cu',
     'tools/data_export.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog, cuda, mpi]
)

executable(
    'cuda_mpi_04',
    ['04/src/main.cpp',
     '04/src/initialization.cu',
     '04/src/simulation.cu',
     'tools/data_export.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog, cuda, mpi]
)
