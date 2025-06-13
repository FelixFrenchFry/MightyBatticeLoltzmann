
project(
    'MightyBatticeLoltzmann',
    ['c', 'cpp', 'cuda'],
    version : '0.1',
    default_options : [
        'warning_level=2',
        # compile specifically for each CUDA architecture
        # potentially used to run the executable
        'cuda_args=' + ' '.join([
            '--std=c++17',
            '-ccbin', '/usr/bin/g++',
            '--ptxas-options=-v',
            '--maxrregcount=240',
            '-gencode', 'arch=compute_61,code=sm_61', # RTX 10s
            '-gencode', 'arch=compute_70,code=sm_70', # V100
            '-gencode', 'arch=compute_75,code=sm_75', # RTX 20s
            '-gencode', 'arch=compute_80,code=sm_80', # A100, A30
            '-gencode', 'arch=compute_86,code=sm_86', # A40, RTX 30s
            '-gencode', 'arch=compute_89,code=sm_89', # RTX 40s
            '-gencode', 'arch=compute_90,code=sm_90'])] # H100, H200
	        #'-gencode', 'arch=compute_100,code=sm_100', # B200, GB200
	        #'-gencode', 'arch=compute_120,code=sm_120'])] # RTX 50s
)

add_project_arguments('-DUSE_FP64', '--std=c++17', language: ['cpp', 'cuda'])

#cuda_include = include_directories('/usr/local/cuda/include')
cuda_include = include_directories('/software/bwhpc/common/toolkit/nvidia_hpc_sdk/25.1/Linux_x86_64/25.1/cuda/include')

cuda = dependency('cuda', required: true)

# detect and integrate MPI
#mpi = dependency('mpi')

#mpi = declare_dependency(
#    compile_args: run_command('/opt/openmpi-cuda-ucx/bin/mpic++', '--showme:compile').stdout().strip().split(),
#    link_args: run_command('/opt/openmpi-cuda-ucx/bin/mpic++', '--showme:link').stdout().strip().split())

#mpi = declare_dependency(
#    compile_args: run_command('/software/bwhpc/common/toolkit/nvidia_hpc_sdk/25.1/Linux_x86_64/25.1/comm_libs/mpi/bin/mpic++', '--showme:compile').stdout().strip().split(),
#    link_args: run_command('/software/bwhpc/common/toolkit/nvidia_hpc_sdk/25.1/Linux_x86_64/25.1/comm_libs/mpi/bin/mpic++', '--showme:link').stdout().strip().split())

# detect and integrate spdlog
spdlog = dependency('spdlog', fallback: ['spdlog', 'spdlog_dep'])

if mpi.found()
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp', 'cuda'])
    message('MPI found')
else
    message('MPI not found, disabling MPI support')
endif

subdir('implementations')
subdir('tests')
