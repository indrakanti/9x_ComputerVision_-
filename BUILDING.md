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

## Running examples

Many existing lessons read images with paths relative to the working directory. Until the lesson cleanup phase is complete, run an example from the directory expected by that lesson or pass an image path when the program supports one.

The build system intentionally separates **compilation correctness** from **lesson runtime data layout**. Upcoming course PRs will standardize runtime arguments and data paths.

## Continuous integration

Pull requests and pushes to `main` run a clean Ubuntu build using `.github/workflows/cmake.yml`.

CI currently validates that the C++ lesson sources compile against the distribution-provided OpenCV 4 development package. Runtime/output regression tests will be added as lessons are standardized.

## Legacy Makefiles

Existing per-directory Makefiles are not removed by the unified build. They remain available during the migration period.

New lessons should use the top-level CMake build so contributors and students have one reproducible path.
