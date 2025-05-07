lib_sources = [  # All source files (excluding headers)
    'hello.cpp'
]

lib = static_library(
    'lib',  # Unique name
    lib_sources,
    dependencies : [mpi]
)

lib_incdirs = include_directories('.')
