# 9x Computer Vision — YouTube Video Series

This file is the public index connecting each YouTube lesson to its source code and notes in this repository.

## Series Positioning

**Computer Vision from First Principles to Production C++ on Linux**

Recommended promise for the playlist:

> Learn computer vision by understanding the math, implementing the ideas in C++, visualizing the results, and then engineering them for real Linux/embedded systems.

## Recommended Episode Format

Aim for focused 12–25 minute lessons rather than one giant course video.

1. **Problem / intuition** — 1–2 min
2. **Whiteboard / math** — 3–7 min
3. **Code walkthrough** — 5–10 min
4. **Run + visualization** — 2–5 min
5. **Engineering note** — 1–3 min
6. **Exercise + next lesson** — <1 min

For larger projects, use 30–45 minute videos.

## Episode Index

Replace `TBD` with the public YouTube URL after each video is published.

| Ep. | Suggested YouTube Title | Repository Topic | YouTube |
|---:|---|---|---|
| 00 | Computer Vision in C++ on Linux — Course Roadmap & Setup | `COURSE_ROADMAP.md` | TBD |
| 01 | How a Camera Becomes Pixels — Light, Sampling & Quantization | `Fundametals/` | TBD |
| 02 | Image Brightness, Contrast, Gamma & Thresholding from First Principles | [lesson + recording script](02_ImageFiltering/01_PointFiltering/VIDEO.md) | TBD |
| 03 | Convolution Explained Visually — The Core Operation of Computer Vision | [lesson + recording script](02_ImageFiltering/00_ConvolutionFirstPrinciples/VIDEO.md) | TBD |
| 04 | Box Filter vs Gaussian Filter — Math, C++ and Real Results | [lesson + recording script](02_ImageFiltering/02_LinearFiltering/00_BoxVsGaussian/VIDEO.md) | TBD |
| 05 | Image Pyramids Explained in C++ — Gaussian, Laplacian, Aliasing & Reconstruction | [lesson + recording script](02_ImageFiltering/03_ImagePyramids/VIDEO.md) | TBD |
| 06 | Sobel & Scharr Image Gradients in C++ — Magnitude, Direction & Signed Derivatives | [lesson + recording script](03_ImageGradients/00_SobelScharr/VIDEO.md) | TBD |
| 07 | Image Derivatives — What Edges Really Are | `05_DerivativesEdgeDetection/` | TBD |
| 08 | Canny Edge Detection from Scratch: NMS + Hysteresis | `05_DerivativesEdgeDetection/` | TBD |
| 09 | Corner Detection — Harris, Shi-Tomasi and FAST | `06_FeatureDetection/` | TBD |
| 10 | SIFT Part 1 — Scale Space and Difference of Gaussians | `07_SIFT/` | TBD |
| 11 | SIFT Part 2 — Keypoint Localization and Orientation | `07_SIFT/` | TBD |
| 12 | SIFT Part 3 — Descriptors and Feature Matching | `07_SIFT/` | TBD |
| 13 | ORB vs SIFT — Speed, Invariance and Embedded Tradeoffs | planned | TBD |
| 14 | Feature Matching Done Correctly — Ratio Test + RANSAC | planned | TBD |
| 15 | Homography Explained — Perspective Transform in Real Applications | planned | TBD |
| 16 | Camera Intrinsics and Extrinsics Without the Confusion | planned | TBD |
| 17 | Camera Calibration in OpenCV C++ + Reprojection Error | planned | TBD |
| 18 | Stereo Vision and Epipolar Geometry — From Two Cameras to Depth | planned | TBD |
| 19 | Optical Flow — Tracking Motion Between Frames | planned | TBD |
| 20 | HOG + SVM Object Detection — Classical Vision Before YOLO | planned | TBD |
| 21 | CNNs for C++ Engineers — Convolution to Classification | planned | TBD |
| 22 | Running ONNX Vision Models from C++ | planned | TBD |
| 23 | YOLO Object Detection in Production C++ | planned | TBD |
| 24 | Segmentation Explained — Semantic vs Instance | planned | TBD |
| 25 | `cv::Mat` Memory Model — Copies, ROIs, Ownership and Performance | planned | TBD |
| 26 | How to Benchmark a Vision Pipeline — Latency, FPS and Jitter | planned | TBD |
| 27 | Multithreaded Computer Vision Pipelines in C++ | planned | TBD |
| 28 | Zero-Copy Computer Vision — What It Means and When It Matters | planned | TBD |
| 29 | Real-Time Linux for Computer Vision — Scheduling and Jitter Basics | planned | TBD |
| 30 | Testing Computer Vision Algorithms with Synthetic Images | planned | TBD |
| 31 | Project: Build a Document Scanner in C++ | planned | TBD |
| 32 | Project: Feature Matching + Panorama Stitching | planned | TBD |
| 33 | Project: Calibrate a Camera and Estimate Pose | planned | TBD |
| 34 | Project: Stereo Depth Pipeline | planned | TBD |
| 35 | Project: Real-Time Object Detection with Latency Metrics | planned | TBD |
| 36 | Capstone: Build a Production-Style C++ Perception Pipeline | planned | TBD |

## Per-Video Repository File

Each lesson should eventually include a `VIDEO.md` containing:

```markdown
# Video: <title>

YouTube: <URL or TBD>

## Learning objectives

## Visual hook / demo

## Script outline

## Equations / diagrams to show

## Code files used

## Commands demonstrated

## Expected output

## Exercise

## References / attribution
```

This makes the repository useful both before and after the video is published.

## YouTube Description Template

```text
In this lesson we learn <topic> from first principles and implement it in C++ on Linux.

Source code + lesson notes:
<GitHub lesson URL>

Full course repository:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- <objective 1>
- <objective 2>
- <objective 3>

Chapters:
00:00 Introduction
...

#ComputerVision #OpenCV #CPP #Linux #EmbeddedAI
```

## Title Strategy

Prefer titles that communicate a concrete engineering outcome rather than generic numbering alone.

Good:
- `Canny Edge Detection from Scratch in C++ — NMS + Hysteresis Explained`
- `cv::Mat Memory Model — Stop Accidentally Copying Images in C++`
- `Camera Calibration in OpenCV C++ — Intrinsics, Distortion & Reprojection Error`

Weak:
- `Computer Vision Tutorial Part 8`
- `OpenCV Lesson 4`

Keep the episode number in the thumbnail/playlist metadata if desired, but let the searchable topic lead the title.

## Shorts / Clips

Each full lesson can produce one 30–60 second short:

- one visual intuition
- one common misconception
- one before/after result
- one performance observation
- one interview-style question

Use the short to point viewers to the complete lesson and repository.
