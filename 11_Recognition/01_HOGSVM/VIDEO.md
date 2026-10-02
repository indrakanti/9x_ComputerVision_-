# Video: HOG + SVM Object Detection — Classical Vision Before YOLO

YouTube: TBD

Suggested title:

**HOG + SVM Object Detection in C++ — Classical Vision Before YOLO**

Suggested thumbnail:

**BEFORE YOLO: HOG + SVM**

Target duration: **32–38 minutes**

## Learning objectives

Viewers should understand:

- gradient magnitude and orientation
- HOG cells and unsigned orientation bins
- interpolated orientation voting
- overlapping block normalization
- why the descriptor is 3780-dimensional
- linear SVM classification
- sliding-window detection
- image pyramids
- non-maximum suppression
- why classical detection differs from modern learned detectors

## Chapters

~~~text
00:00 From shapes to appearance
02:00 Gradients and unsigned orientation
05:00 8x8 HOG cells
09:00 9-bin interpolated voting
12:00 2x2 blocks + L2-Hys
16:00 Why HOG has 3780 features
19:00 Linear SVM intuition
23:00 Train/test split
26:00 Sliding-window detection
29:00 Multi-scale pyramid
32:00 Non-maximum suppression
35:00 Demo + limitations
37:00 Next: classical classifiers
~~~

## Script outline

### 00:00 — From contours to appearance

Show a clean segmented shape from the previous lesson.

Then show a raw image window with clutter.

Ask:

> What if we do not have a perfect segmentation mask first?

Introduce HOG.

### 02:00 — Gradients

Reuse Sobel intuition from earlier modules.

Write:

$$
m=\sqrt{G_x^2+G_y^2}
$$

and:

$$
\theta=\operatorname{atan2}(G_y,G_x)
$$

Explain unsigned 0–180 degree orientation.

### 05:00 — Cells

Overlay an 8x8 grid on the 64x128 window.

Explain one local 9-bin histogram per cell.

### 09:00 — Interpolated voting

Put a gradient angle between two bins.

Split its magnitude vote.

Explain that magnitude weights strong edges more strongly.

### 12:00 — Blocks and normalization

Group 2x2 cells.

Show overlapping block positions.

Explain L2-Hys:

1. normalize
2. clip at 0.2
3. normalize again

### 16:00 — 3780 dimensions

Compute:

~~~text
64x128 window
8x8 cells -> 8x16 cells
2x2 overlapping blocks -> 7x15 positions
36 values per block

7 x 15 x 36 = 3780
~~~

### 19:00 — Linear SVM

Draw positive and negative HOG vectors separated by a plane.

Write:

$$
w^Tx+b=0
$$

Explain margin intuition without turning the lesson into a full optimization course.

### 23:00 — Train/test split

Generate synthetic positives and negatives.

Train the SVM.

Evaluate on a separate random seed.

Emphasize:

> Training accuracy is not generalization.

### 26:00 — Sliding window

Scan one fixed 64x128 classifier over a larger image.

Explain the computational cost.

### 29:00 — Image pyramid

Shrink the image and reuse the same classifier at several scales.

Map the detection rectangles back to original-image coordinates.

### 32:00 — NMS

Show many positive overlapping windows around the same object.

Define IoU.

Keep one representative detection and suppress duplicates.

### 35:00 — Demo

Run:

~~~bash
./build/cv9x_hog_svm \
  --output-dir build/hog_svm
~~~

Show:

- positive sample
- negative sample
- HOG cell visualization
- detection scene
- final boxes

Read holdout accuracy and best synthetic IoU from the console.

### 37:00 — Next

Preview classical classification pipelines with k-NN and SVM over explicit feature vectors.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_hog_svm

./build/cv9x_hog_svm \
  --output-dir build/hog_svm \
  --positives 80 \
  --negatives 80 \
  --test-samples 20 \
  --stride 8 \
  --scale-step 1.20 \
  --max-scales 5

./build/cv9x_hog_svm --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**Why HOG Has 3780 Features**

30–45 seconds:

1. 64x128 window
2. 8x8 cells
3. 2x2 overlapping blocks
4. 9 orientation bins
5. 7 x 15 x 36 = 3780

## YouTube description

This lesson builds a classical object detector from first principles.

We manually compute Histogram of Oriented Gradients, including orientation-bin interpolation and L2-Hys normalization, train a linear SVM, scan an image with a multi-scale sliding window, and suppress duplicate detections with IoU-based non-maximum suppression.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/11_Recognition/01_HOGSVM

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #HOG #SVM #ObjectDetection #Linux
