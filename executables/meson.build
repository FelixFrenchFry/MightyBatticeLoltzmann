
executable(
	'main',
	['main.cpp'],
	include_directories : [lib_incdirs],
	link_with : [lib],
	dependencies : [mpi, spdlog]
)

subdir('02')
subdir('03')
subdir('legacy')
subdir('playground')
subdir('primary')
subdir('singlethreaded')
