
executable(
	'primary',
	['main.cpp', 'density.cu', 'simulation.cu', 'output/export.cpp'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)
