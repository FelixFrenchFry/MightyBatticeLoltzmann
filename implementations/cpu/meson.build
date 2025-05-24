# all exectuables for the CPU implementations

executable(
    'cpu_01',
    ['01/src/main.cpp',
     '01/src/collision.cpp',
     '01/src/conditions.cpp',
     '01/src/density.cpp',
     '01/src/streaming.cpp',
     '01/src/velocity.cpp',
     '01/tools/export.cpp'],
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [spdlog]
)
