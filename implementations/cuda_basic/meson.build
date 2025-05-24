# all exectuables for the single-GPU CUDA implementations

executable(
    'cuda_basic_01',
    ['01/src/main.cpp',
     '01/src/simulation.cu',
     '01/tools/export.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog]
)

executable(
    'cuda_basic_02',
    ['02/src/main.cpp',
     '02/src/collision.cu',
     '02/src/density.cu',
     '02/src/initialization.cu',
     '02/src/streaming.cu',
     '02/src/velocity.cu',
     '02/tools/export.cpp'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog]
)
