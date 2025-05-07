# Meson skeleton code

This repository contains a [Meson](https://mesonbuild.com/) skeleton for C++ projects. It has provision
for dependencies on the numerical libraries

* [Eigen3](https://eigen.tuxfamily.org/)
* [Kokkos](https://kokkos.org/)
* [MPI](https://www.mpi-forum.org/)

## Getting started

Click the `Use this template` button above, then clone the newly created
repository.

### Compiling using CLion

> Note: for Windows users, please follow [these
> instructions](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-product.html)
> in addition to the text below.

If you are using CLion, you can open your freshly cloned project by clicking on
the "Open" button in the CLion welcome window. If prompted, trust the project.

You can change Meson build option in "**File > Settings > Build, Execution, Deployment > Meson**".
This windows allows to set the `buildtype` option, which controls the level of optimization applied to
the code. `debug` disables optimizations and turns on useful debugging features.
This mode should be used when developing and testing code.
`release` turns on aggressive optimization. This mode should be used when
running production simulations. Add `--buildtype=release` or `--buildtype=debug` to "Setup options"
to switch between the two.

To run the executable, click on the dialog directly right of the
green hammer in the upper right toolbar, select "main", and click
the green arrow right of that dialog. You should see the output in the "Run"
tab, in the lower main window.

To run the tests, select "tests" in the same dialog, then run. In the lower
window, on the right, appears a panel that enumerates all the tests that were
run and their results.

Try compiling and running for both `debug` and `release` configurations. Don't
forget to switch between them when testing code or running production simulations.

### Compiling from the command line

The command line (terminal) may look daunting at first, but it has the advantage
of being the same across all UNIX platforms, and does not depend on a specific
IDE. The standard Meson workflow is to create a `builddir/` directory which will
contain all the build files. To do that, and compile your project, run:

```bash
cd <your repository>

# Configure and create build directory
meson setup builddir

# Compile
cd builddir
meson compile

# Run executable and tests
./main
meson test
```

Note that CLion is by default configured to create a `buildDir/` directory
(with a capital `D`).

If there are no errors then you are all set! Note that the flag
`--buildtype=debug` should be changed to
`--buildtype=release` when you run a production simulation, i.e. a
simulation with more than a few hundred atoms. This turns on aggressive compiler
optimizations, which results in speedup. However, when writing the code and
looking for bugs, `debug` should be used instead.

Try compiling and running tests with both compilation configurations.

### Compiling on bwUniCluster, with MPI

The above steps should be done *after* loading the appropriate packages:

```bash
module load compiler/gnu mpi/openmpi

# then in build/
meson setup builddir --buildtype=release
cd builddir
meson compile
```

## How to add code to the repository

There are three places where you are asked to add code:

- `src/` is the core of the code. Code common to all executables and tests
  you will run should be added here. The `meson.build` file in `src/` creates
  [static library](https://en.wikipedia.org/wiki/Static_library) which is linked
  to all the other targets in the repository, and which propagates its dependency,
  so that there is no need to explicitly link against Eigen or MPI.
- `tests/` contains tests for the library code code. It uses
  [GoogleTest](https://google.github.io/googletest/) to define short, simple
  test cases.
- `executables/` contains the final executable codes, i.e. it needs a `main()`
  function.

### Adding to `src/`

Adding files to `src/` is straightforward: create your files, e.g. `lj.h` and
`lj.cpp`, then update the `lib_sources` variable in
`meson.build`:

```meson
lib_sources = [  # All source files (excluding headers)
    'hello.cpp',
    'lj.cpp'
]
```

### Adding to `tests/`

Create your test file, e.g. `test_verlet.cpp` in `tests/`, then modify the
`test_sources` variable in `tests/meson.build`. Test that your test was
correctly added by running `meson compile` in the build directory: your test should
show up in the output.

### Adding to `executables/`

Create a new directory, e.g. with `mkdir executables/04`, then add `subdir('04')`
to `executables/meson.build`, then create & edit
`executables/04/meson.build`:

```meson
executable(
    'milestone04',
    'main.cpp',
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [eigen, mpi]
)
```

You can now create & edit `executables/04/main.cpp`, which should include a
`main()` function as follows:

```c++
int main(int argc, char* argv[]) {
    return 0;
}
```

The code of your simulation goes into the `main()` function.

#### Input files

We often provide input files (`.xyz` files) for your simulations, for example in
milestone 4. You should place these in e.g. `executables/04/`, and add the
following to `executables/04/meson.build`:

```meson
fs = import('fs')
fs.copyfile('lj54.xyz')
```

This will copy the file `executables/04/lj54.xyz` to
`<build>/executables/04/lj54.xyz`, but **only** when the executable for the milestone
is rebuilt. To trigger a rebuild you can erase the `<build>/executables/04`
directory and `meson compile` again.

*Note:* `.xyz` files are ignored by Git. That's on purpose to avoid you staging
very large files in the git tree.

## Pushing code to GitHub

If you have added files to your local repositories, you should commit and push them to
GitHub. To create a new commit (i.e. put your files in the repository's
history), simply run:

```bash
git status
# Look at the files that need to be added
git add <files> ...
git commit -m '<a meaningful commit message!>'
git push
```

This repository is setup with continuous integration (CI), so all your tests
will run automatically when you push. This is very handy to test if you have
breaking changes.

### Git in CLion

If you are using CLion, you can use Git directly from its interface. To add
files, right click the file you wish to add, then "Git > Add". Once you are
ready to commit, "Git > Commit" from the main menu bar. Add a message in the
lower left window where it reads "Commit message", then click "Commit" or
"Commit and Push...".
