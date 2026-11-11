# Part 1 Canonical Course Index

**9x Computer Vision — Classical Computer Vision from First Principles to Production C++ on Linux**

Part 1 is complete at **v1.0.0**.

- Numbered modules: **00–19**
- Video episodes: **00–23**
- Advanced tracking extensions: **A1–A5**
- Next course: [9x ML Computer Vision](https://github.com/indrakanti/9x_ML_ComputerVision)

The module and episode numbers are intentionally not always 1:1. Some modules require multiple focused videos, while projective-geometry foundations are taught together with homography.

| Module | Topic | Canonical repository path | Episode(s) |
|---:|---|---|---:|
| 00 | Course setup, Linux/C++/OpenCV/CMake/CTest/CI | `00_Setup/` | 00 |
| 01 | Image formation, sampling, quantization, color | `01_ImageFormation/` | 01 |
| 02 | Point operations | `02_ImageFiltering/01_PointFiltering/` | 02 |
| 03 | Convolution and linear filtering | `02_ImageFiltering/00_ConvolutionFirstPrinciples/`, `02_ImageFiltering/02_LinearFiltering/` | 03–04 |
| 04 | Image pyramids | `02_ImageFiltering/03_ImagePyramids/` | 05 |
| 05 | Image gradients | `03_ImageGradients/00_SobelScharr/` | 06 |
| 06 | Derivatives and edge detection | `05_DerivativesEdgeDetection/00_DerivativesAndEdges/`, `05_DerivativesEdgeDetection/01_CannyFromScratch/` | 07–08 |
| 07 | Feature detection | `06_FeatureDetection/00_CornerDetectors/` | 09 |
| 08 | Feature descriptors: SIFT/ORB | `07_SIFT/`, `08_FeatureMatching/00_ORBvsSIFT/` | 10–13 |
| 09 | Feature matching and geometric verification | `08_FeatureMatching/01_RatioRansac/` | 14 |
| 10 | Coordinate systems and projective geometry | `09_Geometry/` foundations | 15 (integrated) |
| 11 | Homography and perspective transforms | `09_Geometry/00_HomographyPerspective/` | 15 |
| 12 | Camera model and distortion | `09_Geometry/01_CameraModel/` | 16 |
| 13 | Camera calibration | `09_Geometry/02_CameraCalibration/` | 17 |
| 14 | Stereo vision and epipolar geometry | `09_Geometry/03_StereoVision/` | 18 |
| 15 | Optical flow and motion | `10_Motion/00_OpticalFlow/` | 19 |
| 16 | Contours, components, shape descriptors | `11_Recognition/00_ContoursComponents/` | 20 |
| 17 | HOG + SVM detection | `11_Recognition/01_HOGSVM/` | 21 |
| 18 | Classical classification: k-NN/SVM | `11_Recognition/02_ClassicalClassification/` | 22 |
| 19 | Classical tracking and Kalman basics | `12_Tracking/00_ClassicalTracking/` | 23 |

## Advanced tracking extensions

These deepen Module 19 without changing the numbered course:

| Extension | Topic | Repository path | PR |
|---|---|---|---:|
| A1 | SORT-style multi-object tracking | `12_Tracking/01_MultiObjectTracking/` | 28 |
| A2 | MOG2 + blobs + MeanShift/CamShift | `12_Tracking/02_BackgroundCamShift/` | 29 |
| A3 | CTRV Extended Kalman Filter + NIS/NEES | `12_Tracking/03_ExtendedKalman/` | 30 |
| A4 | MOSSE from first principles + KCF/CSRT comparison | `12_Tracking/04_MOSSECorrelation/` | 31 |
| A5 | Real-video benchmark + success/precision plots | `12_Tracking/04_MOSSECorrelation/REAL_VIDEO_BENCHMARK.md` | 32 |

## Naming compatibility

Historical spelling/casing paths are documented in [NAMING_COMPATIBILITY.md](NAMING_COMPATIBILITY.md). Course-facing names and CMake targets use canonical terminology.

## Part 2

Learned/deep Computer Vision continues in:

**[9x ML Computer Vision](https://github.com/indrakanti/9x_ML_ComputerVision)**

Part 1 remains the classical foundation and should not be renumbered when Part 2 grows.
