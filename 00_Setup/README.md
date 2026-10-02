# Module 00 — Course Setup: Linux, C++17, OpenCV, CMake, CTest & CI

Before learning pixels, filters, features, geometry, or tracking, the development environment must be reproducible.

This module establishes the engineering contract used by every later lesson:

~~~text
source code
   |
   v
CMake configure
   |
   v
C++17 compiler
   |
   v
OpenCV libraries
   |
   v
lesson executable
   |
   v
self-test / CTest
   |
   v
GitHub Actions CI
~~~

The goal is not merely to install OpenCV. The goal is to understand how the repository proves that a lesson builds and runs consistently.

## Learning objectives

By the end of this lesson you should be able to:

- install the Linux build dependencies
- explain the roles of compiler, linker, CMake, OpenCV, and CTest
- configure a CMake build directory
- distinguish source tree from build tree
- build the complete repository or one lesson target
- run one self-test or the full CTest suite
- choose Debug vs Release builds
- enable warnings-as-errors
- understand the clean Ubuntu GitHub Actions workflow
- distinguish configure, compile, link, runtime, and test failures
- use the Module 00 sanity executable to verify a new machine

## 1. Supported baseline

~~~text
Operating system: Linux
Language:         C++17
Build system:     CMake 3.16+
Vision library:   OpenCV 4.x
Tests:            CTest
CI:               GitHub Actions / Ubuntu
~~~

GCC and Clang are both suitable C++ compilers.

## 2. Install dependencies on Ubuntu / Debian

~~~bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev
~~~

build-essential provides the compiler/linker toolchain. CMake generates the build system. pkg-config exposes development-package metadata. libopencv-dev supplies OpenCV headers and libraries.

## 3. Verify tools

~~~bash
g++ --version
cmake --version
pkg-config --modversion opencv4
~~~

Exact versions vary by Linux distribution. The course contract is C++17 and OpenCV 4 or newer.

## 4. Source tree vs build tree

Source:

~~~text
9x_ComputerVision_-/
  CMakeLists.txt
  00_Setup/
  02_ImageFiltering/
  ...
~~~

Build:

~~~text
9x_ComputerVision_-/build/
  CMakeCache.txt
  CMakeFiles/
  executables
  test metadata
  ...
~~~

This is an out-of-source build: generated artifacts stay outside lesson source folders.

## 5. Configure

From the repository root:

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
~~~

Meaning:

~~~text
-S .       source directory is the repository
-B build   generated build tree is build/
Release    optimized build configuration
~~~

Configure is where CMake detects the compiler, checks features, locates OpenCV, creates targets, and registers tests.

## 6. Build everything

~~~bash
cmake --build build --parallel
~~~

The top-level build compiles canonical lessons plus retained legacy examples.

## 7. Build one lesson

Module 00 target:

~~~text
cv9x_course_setup
~~~

Build only it:

~~~bash
cmake --build build --target cv9x_course_setup
~~~

List available targets:

~~~bash
cmake --build build --target help
~~~

## 8. Run environment sanity

~~~bash
./build/cv9x_course_setup \
  --output-dir build/setup
~~~

Outputs:

~~~text
build/setup/setup_sanity.png
build/setup/setup_report.txt
~~~

The report records the C++ language level, OpenCV version, optimization status, OpenCV thread count, timer frequency, image dimensions/type, and a checksum.

The image proves basic OpenCV matrix creation, drawing, color conversion, and image encoding work.

## 9. Run one self-test

~~~bash
./build/cv9x_course_setup --self-test
~~~

The self-test verifies:

- compiler exposes C++17 or newer
- OpenCV major version is 4 or newer
- cv::Mat construction works
- BGR-to-grayscale conversion works
- Gaussian filtering works
- deterministic matrix access/checksum works
- OpenCV high-resolution timing support is available

## 10. Run the whole test suite

~~~bash
ctest --test-dir build --output-on-failure
~~~

CTest executes every registered canonical lesson self-test. The output-on-failure option keeps failing test diagnostics visible.

## 11. What self-tests prove

A lesson self-test should check algorithmic invariants, not merely that the executable launches.

Later examples verify things such as:

- convolution output
- Canny hysteresis
- SIFT scale-space bookkeeping
- homography mapping
- camera reprojection
- stereo depth
- optical-flow motion
- classifier metrics
- tracking lifecycle

## 12. Debug build

~~~bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel
~~~

Debug builds prioritize debuggability and are appropriate for stepping through code, inspecting state, and diagnosing failures.

## 13. Release build

~~~bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
~~~

Release builds enable optimization and are the correct baseline for performance measurements.

Do not benchmark Debug and present it as production performance.

## 14. Compiler warnings

Canonical lesson targets use:

~~~text
-Wall
-Wextra
-Wpedantic
~~~

Warnings are engineering signals, not cosmetic output.

## 15. Warnings as errors

~~~bash
cmake -S . -B build-werror \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON

cmake --build build-werror --parallel
~~~

This converts warnings into build failures for registered targets.

## 16. Failure classes

### Configure failure

Examples:

~~~text
OpenCV not found
compiler unavailable
invalid CMake configuration
~~~

### Compile failure

Examples:

~~~text
syntax error
missing declaration
type mismatch
~~~

### Link failure

Examples:

~~~text
missing library
undefined reference
wrong dependency configuration
~~~

### Runtime failure

Examples:

~~~text
image cannot be loaded
invalid dimensions
bad command-line input
uncaught exception
~~~

### Test failure

The executable runs but violates a required correctness invariant.

Knowing the failure class narrows debugging immediately.

## 17. Shared CMake target helper

The top-level project centralizes lesson setup through:

~~~text
add_cv9x_example(target source)
~~~

Each target receives C++17, OpenCV include paths/libraries, and compiler warnings.

## 18. Stable target names

Some legacy directories contain old spelling or naming inconsistencies.

Course-facing targets remain stable:

~~~text
cv9x_<topic>
~~~

Students should depend on target names rather than legacy path spelling.

## 19. Git workflow

A normal course change follows:

~~~text
main
  |
feature branch
  |
lesson/code/tests
  |
pull request
  |
GitHub Actions
  |
review
  |
merge
~~~

Focused PRs make failures and reviews easier to reason about.

## 20. GitHub Actions CI

For pushes and pull requests targeting main, CI performs:

~~~text
checkout
  |
install compiler/CMake/OpenCV
  |
cmake configure
  |
build all C++ lessons
  |
ctest --output-on-failure
~~~

The clean Ubuntu runner catches machine-specific hidden dependencies.

## 21. Local equivalent of CI

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
~~~

CI then independently repeats the process in a clean environment.

## 22. Reconfigure after build-system changes

When CMakeLists.txt changes:

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
~~~

Explicit reconfiguration keeps the workflow reproducible.

## 23. Clean rebuild

When investigating stale generated state:

~~~bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
~~~

Do not use deletion as a substitute for understanding an actual compiler/test failure.

## 24. Engineering principle

For this course:

> "It works on my machine" is not completion evidence.

A lesson is stronger when dependencies are explicit, the build is reproducible, correctness checks are executable, CI starts clean, and failure output is diagnosable.

## Common setup problems

### OpenCV headers not found

Check:

~~~bash
pkg-config --modversion opencv4
~~~

and confirm libopencv-dev is installed.

### Executable not found

Check:

~~~bash
cmake --build build --target help
~~~

and confirm the target was built.

### Tests report no tests

Ensure the top-level CMake project was configured normally with CTest enabled.

### Code compiled but a lesson test fails

Compilation proves syntax/linkage. The self-test is what checks lesson correctness.

## Exercises

1. Configure separate Debug and Release build directories.
2. Build only cv9x_course_setup.
3. Run its self-test directly and through CTest.
4. Enable warnings-as-errors.
5. Intentionally break one setup assertion and inspect CTest output.
6. Remove the build directory and reproduce a clean build.
7. Map every GitHub Actions step to its local command.
8. Pick one later lesson and identify its CMake target and CTest test name.

## Next lesson

Module 01 starts computer vision itself:

**How light becomes sampled, quantized digital pixels and color values.**

That lesson reorganizes the existing fundamentals material into the canonical course structure.
