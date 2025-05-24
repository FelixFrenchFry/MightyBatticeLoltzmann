
executable(
    'main',
    ['main.cpp'],
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [mpi, spdlog]
)

subdir('cpu')
subdir('cuda_basic')
subdir('cuda_multi')
subdir('milestones')
subdir('playground')
