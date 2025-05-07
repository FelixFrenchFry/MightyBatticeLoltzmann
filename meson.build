# https://mesonbuild.com/
project(
    'my-fantastic-simulation-code', # Project name
    ['c', 'cpp'], # Project type. We need a C++ compiler. The C compiler is required for MPI.
    default_options : ['cpp_std=c++17'], # Yes, we need C++17, at least for constexpr
    version : '0.1'
)

# Optional dependency: Eigen3 linar algebra library
# Make sure to add 'eigen' to the dependencies in your subdirectories if you
# enable this.
# eigen = dependency('eigen3', version: '>=3.4.0')

# Optional dependency: Kokkos C++ performance portability library
# Make sure to add 'kokkoscore' to the dependencies in your subdirectories if you
# enable this.
# kokkoscore = dependency('kokkoscore')

# Detect MPI, but make it optional
mpi = dependency('mpi', language: 'cpp', required: false)

if mpi.found()
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    message('MPI found')
else
    message('MPI not found, disabling MPI support')
endif

subdir('src')
subdir('executables')
subdir('tests')