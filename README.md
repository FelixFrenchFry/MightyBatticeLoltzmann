# MPI-Parallel CUDA Implementation of LBM

## TDLR
I approached the problem in 2 phases:
1. Implement the LBM algo for a single GPU, try out different CUDA techniques to see what works and iteratively optimize the implementation until a performance ceiling is reached. The results can be found in `/implementations/cuda_basic/...`
2. Introduce MPI and domain decomposition to implement LBM for multiple GPUs, and iteratively optimize until there is no motivation left. The results can be found in `/implementations/cuda_mpi/...`

Version `18` is the most important artifact of phase 1 and a variety of benchmarks (different domain sizes, GPU models, etc...) can be found in its sub-folder.<br><br>
Version `04` is the most important artifact of phase 2 and a variety of benchmarks can be found in its sub-folder. Peak performance of 1101 BLUPS across 40 H100 GPUs (96% of ideal value assuming linear scaling with 28.6 BLUPS per H100).<br><br>
The output (.out/.err) of various job scripts for the HPC cluster can be found in `/job_scripts_output`. The most notable results are summarized in the benchmark files in the respective implementations' sub-folders.<br><br>
Plotted simulation results can be found in `/implementations/cuda_basic/tools/output/...` and `/implementations/cuda_mpi/tools/output/...` for some of the single- and multi-GPU implementations, respectively.

## Summaries of the different implementations along the way

### `/implementations/cuda_basic/...`

* `01` - first attempt with sequential, non-coalesced memory transfers and separate kernels for each operation (1. density compute, 2. velocity compute, 3. collision, 4. streaming)
* `02` - like version 01, but with coalesced memory transfers
* `03` - like version 02, but with attempts to populate shared memory with DF values and read from there
* `04` - like version 03, but with fused density/velocity and collision/streaming kernels for lower total number of global memory loads (BIG GAIN)
* `05` - like version 04, but with fully fused kernel for all operations (BIG GAIN)
* `06` - like version 05, but with attempts to use explicitly vectorized memory transfers via custom struct for the DF values (unfinished, does not compile)
* `07` - like version 05, but with explicitly unrolled loops
* `08` - like version 05, but with pull instead of push streaming and without shared memory usage
* `09` - like version 08, but with attempts to populate shared memory with DF values (unfinished, does not compile)
* `10` - like version 05, but without global writes to store density/velocity values and without shared memory usage (BIG GAIN)
* `11` - like version 10, but with sequential, non-coalesced memory transfers (BIG DROP)
* `12` - like version 10, but with flattened 1D array instead of 2D array for the coalesced memory transfers
* `13` - like version 10, but with inlined device kernels for better organization, optional global writes to store density/velocity values, and new lid driven cavity simulation mode
* `14` - like version 13, but with FP32/FP64 selection at compile time
* `15` - like version 14, but with attempts to micro-optimize the computations
* `16` - like version 15, but with pull instead of push streaming and re-introduction of populating shared memory for DF values
* `17` - like version 16, but without loop unrolling compiler hints
* `18` - like version 17, but with QOL improvements regarding data export and progress display
* `19` - like version 18, but with QOL improvements such as optional input file for simulation parameters

### `/implementations/cuda_mpi/...`

* `01` - first attempt at turning cuda_basic_18 into an MPI-parallel version with 1D domain decomposition along the Y-axis and separate arrays for halo DF values
* `02` - like version 01, but with separate kernels for inner/outer cells
* `03` - like version 02, but with QOL improvements such as optional input file for simulation parameters
* `04` - like version 03, but with async MPI communication to overlap compute of inner cells and exchange of halo cells 
* `05` - like version 04, but with attempts to do 2D domain decomposition instead of 1D (unfinished, does not compile)
* `06` - like version 04, but without optional "branchless" sub-kernels and cleanup of unused leftover code
* `07` - zombie version of 04, used for presentation screenshots of simplified and compressed code (does definitely not compile and is not supposed to either)
