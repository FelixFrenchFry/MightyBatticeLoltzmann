
executable(
	'simulation_vis',
	['main.cpp', 'simulation.cu', 'exporter.cpp'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)
