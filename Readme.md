# 9x Computer Vision

## Computer Vision from First Principles to Production C++ on Linux

9x Computer Vision is an open learning series for engineers who want to understand **how computer vision works**, not only how to call an OpenCV API.

Each lesson connects four layers:

1. **Theory and mathematics** — the physical or mathematical idea behind the algorithm.
2. **C++ implementation** — practical code on Linux, primarily using OpenCV.
3. **Visualization** — intermediate steps, output images, and parameter effects.
4. **Engineering** — performance, memory behavior, failure cases, testing, and deployment considerations.

Python examples are included where they improve accessibility or make comparison useful, but the primary engineering track is C++.

## Course Resources

- [Complete Course Roadmap](COURSE_ROADMAP.md)
- [YouTube Video Series and Link Index](VIDEO_SERIES.md)
- [Linux Build Guide](BUILDING.md)
- [Lesson Authoring Guide](LESSON_GUIDE.md)

The video index is designed so every published YouTube lesson can link directly to its corresponding theory, source code, commands, and exercises in this repository.

## Current Repository Content

The repository already contains material covering:

- image formation / computer-vision fundamentals
- point filtering and intensity transformations
- linear filtering and Gaussian filtering
- image gradients
- image blending
- derivatives and edge detection
- feature detection
- SIFT concepts and implementation exercises

The canonical classical course now covers:

- environment setup and reproducible C++/OpenCV builds
- image formation, sampling, quantization, and color representation
- filtering, pyramids, gradients, edges, and corners
- SIFT/ORB descriptors and feature matching
- RANSAC, homography, camera models, calibration, stereo, and metric depth
- optical flow and temporal feature tracking
- contours, connected components, and shape descriptors
- HOG + SVM detection
- k-NN/SVM classical classification
- correlation + Kalman object tracking and lifecycle states

**Modules 00–19 / Episodes 00–23 are the completed numbered classical Computer Vision course.**

Learned/deep Computer Vision is intentionally continuing in a separate **ML Computer Vision** repository. Production C++ topics and applied projects may be revisited later as optional extensions.

## Learning Pattern

A completed lesson should answer four questions:

> **What is happening mathematically?**
>
> **How do I implement it in C++?**
>
> **How do I prove the output is correct?**
>
> **What changes when I put it into a real-time or embedded vision system?**

That final question is an important part of the 9x Computer Vision direction.

## Build Environment

The C++ examples target Linux, C++17, and OpenCV 4. A top-level CMake build is the canonical build path.

Typical dependencies on Ubuntu are:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev
```

Configure and build all current C++ lessons from the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Build a single lesson by its stable target name:

```bash
cmake --build build --target cv9x_sobel
cmake --build build --target cv9x_sift_config_match
```

See [BUILDING.md](BUILDING.md) for debug builds, warnings-as-errors, runtime notes, and the migration policy for legacy Makefiles.

Pull requests and pushes to `main` are compiled on Ubuntu through GitHub Actions so build regressions can be caught before course material is published.

## Repository Naming and Lesson Structure

Some early lesson directories contain legacy spelling or naming inconsistencies. Those paths are kept temporarily so linked images and lesson references are not broken during the build modernization.

New lessons use normalized names and stable CMake targets. In particular, course-facing build targets use the correct **SIFT** terminology even where a legacy path currently contains `SHIFT`.

See [LESSON_GUIDE.md](LESSON_GUIDE.md) for the standard lesson layout and definition of done.

## Course Status

| Stage | Focus | Status |
|---|---|---|
| v0.1 | Classical foundations, Modules 00–09 | Complete |
| v0.2 | Geometry, cameras, stereo and motion, Modules 10–15 | Complete |
| v0.3 | Recognition and tracking, Modules 16–19 | Complete |
| v1.0 | Complete numbered classical course, Modules 00–19 | Complete |

The active numbered series ends here. ML/deep Computer Vision continues in a separate repository so classical and learned perception remain cleanly separated.

## YouTube Integration

Every lesson will have a corresponding entry in [VIDEO_SERIES.md](VIDEO_SERIES.md). When a video is published, its URL can be added to the table so GitHub and YouTube become two views of the same course:

```text
YouTube lesson
      |
      v
Theory + diagrams
      |
      v
C++ source code
      |
      v
Build / run commands
      |
      v
Results + exercises
```

## Contributions

Contributions that improve explanations, fix code, add tests, create useful visualizations, or provide portable build support are welcome.

New or modernized lessons should follow [LESSON_GUIDE.md](LESSON_GUIDE.md) and build through the top-level CMake project.

A repository license still needs to be selected before the project is presented as a fully reusable open-source course.

## Course Philosophy

This repository should not become a collection of OpenCV one-liners.

The goal is to help an engineer understand a vision algorithm from **light and pixels**, through **math and implementation**, all the way to **runtime behavior and deployment**.
