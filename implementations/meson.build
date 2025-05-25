# subdirectory overview

lib = static_library(
    'lib',
    dependencies : [mpi, spdlog]
)

lib_incdirs = include_directories('.')

subdir('cpu')
subdir('cuda_basic')
subdir('cuda_multi_gpu')
subdir('cuda_multi_node')
subdir('milestones')
subdir('playground')
