
executable(
	'milestone03',
	['main.cpp', 'collision.cu'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)