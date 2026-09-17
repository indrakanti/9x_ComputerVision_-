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
| 00 | Course setup: Linux, C++, OpenCV, CMake, repo workflow | Build foundation complete; lesson planned |
| 01 | Light, cameras, pixels, sampling, quantization, color spaces | Existing material; reorganize |
| 02 | Point operations: brightness, contrast, inversion, gamma, thresholding | **Video-ready reference lesson** |
| 03 | Linear filtering and convolution: manual convolution, box and Gaussian filters | **Video-ready: convolution + box/Gaussian** |
| 04 | Image pyramids and multi-scale processing | **Video-ready: Gaussian + Laplacian + reconstruction** |

### Part II — Gradients, Edges, and Features

| Module | Topic | Status |
|---|---|---|
| 05 | Image gradients: Sobel, Scharr, magnitude, direction | Existing |
| 06 | Derivatives and edge detection: Laplacian, Canny, NMS, hysteresis | Existing / expand |
| 07 | Feature detection: Harris, Shi-Tomasi, FAST | Existing / expand |
| 08 | Feature descriptors: SIFT, ORB, BRIEF | SIFT existing; ORB/BRIEF planned |
| 09 | Feature matching: BF, FLANN, ratio test, geometric verification | Planned |

### Part III — Geometry and Cameras

| Module | Topic | Status |
|---|---|---|
| 10 | Coordinate systems and projective geometry | Planned |
| 11 | Homography and perspective transforms | Planned |
| 12 | Camera model, intrinsic/extrinsic parameters, distortion | Planned |
| 13 | Camera calibration and reprojection error | Planned |
| 14 | Stereo vision, epipolar geometry, disparity, depth | Planned |
| 15 | Optical flow and motion estimation | Planned |

### Part IV — Classical Recognition and Tracking

| Module | Topic | Status |
|---|---|---|
| 16 | Contours, connected components, shape descriptors | Planned |
| 17 | HOG + SVM object detection | Planned |
| 18 | Classical classification: k-NN, SVM, feature pipelines | Planned |
| 19 | Tracking: template matching, correlation, Kalman-filter basics | Planned |

### Part V — Deep Computer Vision

| Module | Topic | Status |
|---|---|---|
| 20 | CNN fundamentals from the convolution operation upward | Planned |
| 21 | Classification inference with OpenCV DNN / ONNX Runtime | Planned |
| 22 | Object detection: YOLO-family inference concepts | Planned |
| 23 | Semantic and instance segmentation | Planned |
| 24 | Pose / keypoint estimation | Planned |
| 25 | Vision transformers and modern feature representations | Planned |

### Part VI — Production C++ Vision on Linux

This part is the primary differentiator of the course.

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

### Part VII — Applied Projects

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

1. **Done:** add a top-level CMake build.
2. Standardize naming and fix spelling/casing inconsistencies.
3. **Done:** add CI to compile the C++ examples and run lesson tests.
4. Add formatting/static-analysis configuration.
5. Add a license selected by the repository owner.
6. Add contribution guidelines.
7. Add datasets/assets attribution where required.
8. Add a release/tagging convention matching YouTube course milestones.

## Recommended Release Milestones

- **v0.1 — Classical Vision Foundations:** Modules 00–09
- **v0.2 — Geometry and Cameras:** Modules 10–15
- **v0.3 — Recognition and Tracking:** Modules 16–19
- **v0.4 — Deep Vision:** Modules 20–25
- **v0.5 — Production C++ Vision:** Modules 26–35
- **v1.0 — Complete Course:** all modules + applied projects + capstone

## Course Principle

The goal is not to become another collection of OpenCV one-liners. The course should teach learners to answer:

> What happens mathematically, what happens in memory and at runtime, how do I know it works, and how would I deploy it in a real C++ vision system?
