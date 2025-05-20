
executable(
	'main',
	['main.cpp'],
	include_directories : [lib_incdirs],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)

subdir('02')
subdir('playground')
subdir('simulation_rendering')
