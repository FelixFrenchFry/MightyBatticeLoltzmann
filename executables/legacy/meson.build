
executable(
	'legacy',
	['main.cpp', 'simulation.cu', 'output/export.cpp'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)
