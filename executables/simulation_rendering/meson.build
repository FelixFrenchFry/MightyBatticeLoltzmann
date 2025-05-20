
executable(
	'simulation_rendering',
	['main.cpp', 'simulation.cu'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)
