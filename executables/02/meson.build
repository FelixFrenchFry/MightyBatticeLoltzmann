
executable(
	'milestone02',
	['main.cpp', 'streaming.cu'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)