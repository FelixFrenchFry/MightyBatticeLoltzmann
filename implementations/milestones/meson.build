# all exectuables for the implementations of the lecture notes milestones

executable(
    'milestone_02',
    ['02/src/main.cpp',
     '02/src/streaming.cu'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog]
)

executable(
    'milestone_03',
    ['03/src/main.cpp',
     '03/src/collision.cu'],
    include_directories : [lib_incdirs, cuda_include],
    link_with : [lib],
    dependencies : [spdlog]
)
