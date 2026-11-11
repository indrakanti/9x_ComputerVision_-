# Building 9x Computer Vision on Linux

The repository now has one top-level CMake build for the current C++ lesson examples.

## Supported baseline

- Linux
- C++17 compiler (GCC or Clang)
- CMake 3.16+
- OpenCV 4.x development packages

## Ubuntu / Debian dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

## First-machine sanity check

From a clean checkout, configure, build, and run Module 00:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_course_setup
./build/cv9x_course_setup --self-test
./build/cv9x_course_setup --output-dir build/setup
```

This verifies the C++17/OpenCV environment and writes a small environment report plus sanity image.

See [00_Setup/README.md](00_Setup/README.md) for the teaching walkthrough.

## Configure and build everything

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Executables are generated inside the `build/` tree.

## Build one lesson

CMake targets use stable names even when an older source directory contains a spelling inconsistency.

Examples:

```bash
cmake --build build --target cv9x_sobel
cmake --build build --target cv9x_feature_detection
cmake --build build --target cv9x_sift_config_match
```

To see all targets:

```bash
cmake --build build --target help
```

## Debug build

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel
```

## Warnings as errors

Course development can optionally make compiler warnings fatal:

```bash
cmake -S . -B build-werror \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON
cmake --build build-werror --parallel
```

This is intentionally opt-in while older examples are being modernized.

## Formatting

The repository includes a root `.clang-format`. Format a changed C++ file with:

```bash
clang-format -i path/to/file.cpp
```

For a patch, prefer formatting only the files you changed rather than reformatting unrelated legacy code.

## clang-tidy

The repository includes a pragmatic root `.clang-tidy`. It is intentionally opt-in so contributors are not required to install LLVM tooling for the baseline build.

Run clang-tidy through CMake:

```bash
cmake -S . -B build-tidy \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCV9X_ENABLE_CLANG_TIDY=ON
cmake --build build-tidy --parallel
```

The configuration emphasizes bug-prone, performance, portability, and a small set of low-noise readability checks.

## Editor defaults

A root `.editorconfig` standardizes UTF-8, LF line endings, final newlines, and four-space C++ indentation.

## Running examples

Canonical lessons use explicit command-line arguments, deterministic synthetic inputs where appropriate, and `--self-test` modes registered with CTest.

Some retained legacy examples still read images relative to their historical working directories. Those examples remain available during migration, but the canonical lesson path is the top-level CMake + CTest workflow.

## Continuous integration

Pull requests and pushes to `main` run a clean Ubuntu build using `.github/workflows/cmake.yml`.

CI validates three stages on a clean Ubuntu runner:

```text
CMake configure
full C++ build
CTest lesson self-tests
```

The canonical lesson self-tests validate deterministic algorithmic invariants in addition to compilation correctness.

## Run the complete lesson test suite

```bash
ctest --test-dir build --output-on-failure
```

This is the same test stage used by GitHub Actions.

## Legacy Makefiles

Existing per-directory Makefiles are not removed by the unified build. They remain available during the migration period.

New lessons should use the top-level CMake build so contributors and students have one reproducible path.
