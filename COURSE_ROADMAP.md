# 9x Computer Vision — Course Roadmap

## Course Positioning

**Computer Vision from first principles to production C++ on Linux.**

The course is designed for engineers who want more than API-level OpenCV examples. Each module should connect:

1. **Theory and mathematics** — what the algorithm is doing and why.
2. **Implementation** — readable C++ first, with Python references where useful.
3. **Visualization** — images, plots, intermediate stages, and failure cases.
4. **Engineering** — runtime cost, memory behavior, numerical issues, interfaces, and testability.
5. **Application** — a small real-world exercise or project.

## Target Audience

- Software engineers moving into computer vision
- Embedded / robotics / automotive developers
- Students who know C/C++ and want to understand vision beyond Python notebooks
- Engineers preparing for perception, robotics, ADAS, autonomous-system, and edge-AI roles

## Course Structure

### Part I — Image Formation and Fundamentals

| Module | Topic | Status |
|---|---|---|
| 00 | Course setup: Linux, C++, OpenCV, CMake, repo workflow | **Video-ready: toolchain + CMake + CTest + CI workflow** |
| 01 | Light, cameras, pixels, sampling, quantization, color spaces | **Video-ready: sampling + aliasing + quantization + color representation** |
| 02 | Point operations: brightness, contrast, inversion, gamma, thresholding | **Video-ready reference lesson** |
| 03 | Linear filtering and convolution: manual convolution, box and Gaussian filters | **Video-ready: convolution + box/Gaussian** |
| 04 | Image pyramids and multi-scale processing | **Video-ready: Gaussian + Laplacian + reconstruction** |

### Part II — Gradients, Edges, and Features

| Module | Topic | Status |
|---|---|---|
| 05 | Image gradients: Sobel, Scharr, magnitude, direction | **Video-ready: signed gradients + magnitude/direction** |
| 06 | Derivatives and edge detection: Laplacian, Canny, NMS, hysteresis | **Video-ready: derivatives/Laplacian + Canny from scratch** |
| 07 | Feature detection: Harris, Shi-Tomasi, FAST | **Video-ready: structure tensor + three corner detectors** |
| 08 | Feature descriptors: SIFT, ORB, BRIEF | **Video-ready: complete SIFT + ORB/SIFT engineering comparison** |
| 09 | Feature matching: BF, FLANN, ratio test, geometric verification | **Video-ready: ratio test + manual homography RANSAC verification** |

### Part III — Geometry and Cameras

| Module | Topic | Status |
|---|---|---|
| 10 | Coordinate systems and projective geometry | **Video-ready foundation: homogeneous coordinates + projective point mapping** |
| 11 | Homography and perspective transforms | **Video-ready: manual 8-DoF solve + inverse warp + rectification** |
| 12 | Camera model, intrinsic/extrinsic parameters, distortion | **Video-ready: manual 3-D→2-D projection + lens distortion** |
| 13 | Camera calibration and reprojection error | **Video-ready: synthetic calibration + residual diagnostics** |
| 14 | Stereo vision, epipolar geometry, disparity, depth | **Video-ready: E/F + rectification + disparity + metric depth** |
| 15 | Optical flow and motion estimation | **Video-ready: Lucas-Kanade + pyramids + temporal feature tracks** |

### Part IV — Classical Recognition and Tracking

| Module | Topic | Status |
|---|---|---|
| 16 | Contours, connected components, shape descriptors | **Video-ready: region labeling + contour topology + descriptors** |
| 17 | HOG + SVM object detection | **Video-ready: manual HOG + linear SVM + multiscale detection** |
| 18 | Classical classification: k-NN, SVM, feature pipelines | **Video-ready: scaling + k-NN/RBF-SVM + confusion metrics** |
| 19 | Tracking: template matching, correlation, Kalman-filter basics | **Video-ready: NCC + Kalman prediction/correction + lifecycle states** |

### Classical Course Boundary

**Modules 00–19 are the completed scope of this repository's numbered classical Computer Vision course.**

Advanced tracking A1–A5 and repository closeout are tracked in [ADVANCED_TRACKING_ROADMAP.md](ADVANCED_TRACKING_ROADMAP.md). The canonical module/episode/path mapping is in [COURSE_INDEX.md](COURSE_INDEX.md).

Learned/deep Computer Vision topics—CNNs, ONNX inference, YOLO-family detection, learned segmentation/pose, and vision transformers—continue in **[9x ML Computer Vision](https://github.com/indrakanti/9x_ML_ComputerVision)** rather than extending the numbered sequence here.

### Deferred Ideas — Not Part of the v1.0 Numbered Course

The following production-C++ topics remain useful future material, but they are explicitly **not required for Part 1 completion** and are not part of Modules 00–19.

| Module | Topic | Status |
|---|---|---|
| 26 | CMake project structure, libraries, tests, sanitizers | Planned |
| 27 | cv::Mat memory model, ownership, copies, ROIs, alignment | Planned |
| 28 | Performance measurement: latency, throughput, FPS, percentiles | Planned |
| 29 | Multithreading and bounded pipelines | Planned |
| 30 | Zero-copy concepts and camera-buffer interfaces | Planned |
| 31 | SIMD, OpenCV optimization paths, GPU/accelerator overview | Planned |
| 32 | Real-time considerations: scheduling, jitter, memory allocation | Planned |
| 33 | Robustness: malformed inputs, numerical limits, error handling | Planned |
| 34 | Testing vision algorithms with synthetic images and golden data | Planned |
| 35 | Deployment: ONNX, containers, embedded/edge devices | Planned |

### Optional Applied Projects

| Project | Deliverable |
|---|---|
| P1 | Document scanner using edges + contours + homography |
| P2 | Feature-based image matching and panorama stitching |
| P3 | Calibrate a camera and estimate pose |
| P4 | Stereo depth pipeline |
| P5 | Lane / road-feature perception pipeline |
| P6 | Real-time object detector with latency instrumentation |
| P7 | Multi-stage Linux C++ perception pipeline with bounded queues |
| Capstone | Production-style perception application with tests, metrics, configuration, and documented failure modes |

## Standard Lesson Layout

Every lesson should use the same structure:

```text
README.md             theory, equations, diagrams, expected output
src/                  C++ implementation
python/               optional Python reference
include/               reusable headers where appropriate
assets/                input/output images and diagrams
tests/                 unit or regression tests
CMakeLists.txt         build definition
VIDEO.md               video script/outline and YouTube link
```

Not every introductory exercise needs all folders, but advanced modules should converge on this structure.

## Definition of Done for a Lesson

A lesson is complete when it has:

- learning objectives
- prerequisite list
- theory and equations
- at least one C++ implementation
- reproducible build/run commands
- expected output
- explanation of important parameters
- common failure cases
- one exercise for the learner
- a YouTube lesson entry in `VIDEO_SERIES.md`
- tested code on Linux

## Repository Modernization Work

1. **Done:** top-level CMake build.
2. **Done:** canonical naming plus compatibility documentation for legacy paths.
3. **Done:** CI compiles the C++ examples and runs lesson tests.
4. **Done:** clang-format, clang-tidy, EditorConfig, and PR hygiene.
5. **Done:** Apache License 2.0.
6. **Done:** contribution guidelines.
7. **Done:** explicit external-data/media licensing policy and reproducible-download guidance.
8. **Done:** v1.0.0 release convention and canonical course index.

Broader multi-platform/compiler CI and deeper modernization of every historical example are maintenance opportunities, not Part 1 release blockers.

## Release Milestones

- **v0.1 — Classical Vision Foundations:** Modules 00–09
- **v0.2 — Geometry, Cameras, Stereo and Motion:** Modules 10–15
- **v0.3 — Recognition and Tracking:** Modules 16–19
- **v1.0 — Classical Course Complete:** Modules 00–19, all canonical lessons build and self-test in CI

Production-C++ topics and applied projects remain optional extensions. They do not block completion of the classical numbered course.

## Course Principle

The goal is not to become another collection of OpenCV one-liners. The course should teach learners to answer:

> What happens mathematically, what happens in memory and at runtime, how do I know it works, and how would I deploy it in a real C++ vision system?
