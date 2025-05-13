
project(
    'MightyBatticeLoltzmann',
    ['c', 'cpp', 'cuda'],
    version : '0.1',
    default_options : [
        'cpp_std=c++17', # use C++20 instead?
        'cuda_std=c++17',
        'warning_level=3',
        # compile specifically for each CUDA architecture
        # potentially used to run the executable
        'cuda_args=' + ' '.join([
            '-ccbin', '/usr/bin/g++-12',
            '-gencode', 'arch=compute_75,code=sm_75', # RTX 20s
            '-gencode', 'arch=compute_80,code=sm_80', # A30, A100
            '-gencode', 'arch=compute_86,code=sm_86', # A40, RTX 30s
            '-gencode', 'arch=compute_89,code=sm_89', # RTX 40s
            '-gencode', 'arch=compute_90,code=sm_90', # H100, H200
            '-gencode', 'arch=compute_100,code=sm_100', # B200, GB200
            '-gencode', 'arch=compute_120,code=sm_120']), # RTX 50s
        # specify location of used CUDA libraries
        'link_args=' + ' '.join([
            '-L/usr/local/cuda/lib64', '-lcudart'])]
)

# addresssanitizer flags for detecting runtime memory errors
# add_project_arguments('-fsanitize=address', language: ['c', 'cpp', 'cuda'])
# add_project_link_arguments('-fsanitize=address', language: ['c', 'cpp', 'cuda'])

# optional dependency: Kokkos C++ performance portability library
# make sure to add 'kokkoscore' to the dependencies in your subdirectories
# kokkoscore = dependency('kokkoscore')

# detect MPI, but make integration optional
mpi = dependency('mpi', language: 'cpp', required: false)

# detect and integrate SDL2
# sdl2 = dependency('sdl2', fallback: ['sdl2', 'sdl2_dep'])

# detect and integrate spdlog
spdlog = dependency('spdlog', fallback: ['spdlog', 'spdlog_dep'])

if mpi.found()
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp', 'cuda'])
    message('MPI found')
else
    message('MPI not found, disabling MPI support')
endif

subdir('src')
subdir('executables')
subdir('tests')
