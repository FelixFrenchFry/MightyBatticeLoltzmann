
executable(
	'primary',
	['main.cpp', 'collision.cu', 'density.cu', 'streaming.cu', 'velocity.cu', 'output/export.cpp'],
	include_directories : [lib_incdirs, cuda_include],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)
