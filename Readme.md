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

The roadmap extends this foundation into:

- feature matching and geometric verification
- homography and projective geometry
- camera calibration
- stereo vision and depth
- optical flow and tracking
- classical object detection and classification
- CNNs, ONNX inference, object detection, segmentation, and pose
- production C++ topics such as memory ownership, benchmarking, multithreading, zero-copy concepts, real-time considerations, testing, and deployment
- end-to-end projects and a capstone perception pipeline

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

The examples target Linux and use OpenCV. Existing lessons currently use local Makefiles in several directories. A top-level CMake build and CI are planned as part of the course modernization work.

Typical dependencies on Ubuntu are:

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libopencv-dev
```

Exact build instructions remain inside each lesson until the unified build is introduced.

## Course Stages

| Stage | Focus |
|---|---|
| v0.1 | Classical vision foundations |
| v0.2 | Geometry, cameras, calibration, stereo |
| v0.3 | Recognition and tracking |
| v0.4 | Deep computer vision |
| v0.5 | Production C++ vision on Linux |
| v1.0 | Complete course + projects + capstone |

See [COURSE_ROADMAP.md](COURSE_ROADMAP.md) for the module-by-module plan.

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

A formal contribution guide and repository license are still to be added.

## Course Philosophy

This repository should not become a collection of OpenCV one-liners.

The goal is to help an engineer understand a vision algorithm from **light and pixels**, through **math and implementation**, all the way to **runtime behavior and deployment**.
