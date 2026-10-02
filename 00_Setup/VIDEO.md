# Video: Computer Vision in C++ on Linux — Course Setup & Engineering Workflow

YouTube: TBD

Suggested title:

**Computer Vision in C++ on Linux — OpenCV, CMake, CTest & CI Setup**

Suggested thumbnail:

**BUILD ONCE. RUN EVERYWHERE.**

Target duration: **22–28 minutes**

## Learning objectives

Viewers should understand:

- the Linux/C++17/OpenCV baseline
- what CMake configures
- source tree vs build tree
- how to build all lessons or one target
- how lesson self-tests work
- how CTest runs the course test suite
- Debug vs Release
- compiler warnings
- warnings-as-errors
- how GitHub Actions reproduces the build on clean Ubuntu

## Chapters

~~~text
00:00 What this course expects
01:30 Install compiler, CMake and OpenCV
04:00 Verify the toolchain
06:00 Source tree vs build tree
08:00 Configure with CMake
10:00 Build everything vs one target
12:30 Run the Module 00 sanity executable
15:00 Self-tests and CTest
17:30 Debug vs Release
20:00 Warnings and warnings-as-errors
22:00 Git workflow + GitHub Actions
25:00 Next: how light becomes pixels
~~~

## Script outline

### 00:00 — Course contract

Start with the course promise:

> We will not treat OpenCV calls as magic. Every lesson needs math, implementation, evidence, and engineering context.

Show the repository and the top-level build.

### 01:30 — Install dependencies

Run:

~~~bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev
~~~

Explain what each package contributes.

### 04:00 — Verify the environment

Run:

~~~bash
g++ --version
cmake --version
pkg-config --modversion opencv4
~~~

Explain that exact versions can differ, while the course contract is C++17 + OpenCV 4.x.

### 06:00 — Source vs build tree

Show:

~~~text
source code
build/
~~~

Explain why generated files stay out of lesson directories.

### 08:00 — Configure

Run:

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
~~~

Explain:

- source path
- build path
- build type
- dependency detection
- target registration
- CTest registration

### 10:00 — Build

Build everything:

~~~bash
cmake --build build --parallel
~~~

Then build one target:

~~~bash
cmake --build build --target cv9x_course_setup
~~~

### 12:30 — Sanity executable

Run:

~~~bash
./build/cv9x_course_setup \
  --output-dir build/setup
~~~

Open:

~~~text
setup_sanity.png
setup_report.txt
~~~

Show the C++ language level and OpenCV version.

### 15:00 — Self-test and CTest

Run:

~~~bash
./build/cv9x_course_setup --self-test
~~~

Then:

~~~bash
ctest --test-dir build --output-on-failure
~~~

Explain why compilation alone is not correctness evidence.

### 17:30 — Debug vs Release

Configure separate directories.

Debug:

~~~bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug
~~~

Release:

~~~bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release
~~~

Explain why performance numbers must come from an optimized build.

### 20:00 — Warnings

Show:

~~~text
-Wall
-Wextra
-Wpedantic
~~~

Then enable:

~~~bash
cmake -S . -B build-werror \
  -DCMAKE_BUILD_TYPE=Release \
  -DCV9X_WARNINGS_AS_ERRORS=ON
~~~

### 22:00 — PR workflow and CI

Draw:

~~~text
branch
  -> commit
  -> PR
  -> clean Ubuntu CI
  -> configure
  -> build
  -> CTest
  -> merge
~~~

Open the GitHub Actions workflow and map each step to the local command already demonstrated.

### 25:00 — Next

Say:

> The environment is now proven. In Episode 01 we start at the physical beginning: light, sampling, quantization, pixels, and color.

## Commands demonstrated

~~~bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev

g++ --version
cmake --version
pkg-config --modversion opencv4

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --build build --target cv9x_course_setup

./build/cv9x_course_setup \
  --output-dir build/setup

./build/cv9x_course_setup --self-test

ctest --test-dir build --output-on-failure
~~~

## Short idea

**Compile Success Does NOT Mean Your Vision Algorithm Is Correct**

30–45 seconds:

1. code compiles
2. executable launches
3. self-test checks a known invariant
4. CTest runs all lesson invariants
5. close with: "build evidence and algorithm evidence are different"

## YouTube description

This is the engineering foundation for the 9x Computer Vision course.

We set up C++17, OpenCV, CMake and CTest on Linux, build the repository, run a deterministic environment sanity check, and show how GitHub Actions reproduces the same configure/build/test workflow on a clean Ubuntu runner.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/00_Setup

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #CMake #Linux #CTest
