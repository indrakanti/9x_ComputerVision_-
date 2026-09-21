# Video: Image Derivatives & Edge Detection — First Derivative, Laplacian and Zero Crossings

YouTube: TBD

Suggested title:

**Image Derivatives & Edge Detection in C++ — Sobel, Laplacian & Zero Crossings**

Suggested thumbnail:

**WHERE IS THE EDGE?**

Target duration: **18–24 minutes**

## Learning objectives

Viewers should understand:

- why edges are spatial changes
- first- vs second-derivative edge evidence
- gradient-magnitude thresholding
- the signed Laplacian
- zero-crossing intuition
- why smoothing matters before differentiation
- why visualization normalization must not define algorithm thresholds

## Visual hook

Start with a 1-D dark-to-bright step.

Animate three rows:

```text
Intensity          ______|------
1st derivative          /\
2nd derivative        +  -
                         ^
                     sign change
```

Then show the same idea on a real/synthetic image:

- gradient magnitude
- thresholded edge map
- signed Laplacian
- zero-crossing map

Narration:

> We can detect an edge by looking for a large first derivative, or by looking for where the second derivative changes sign. Those are related ideas, but they are not the same algorithm.

## Chapters

```text
00:00 What is an edge mathematically?
01:40 Step edge and the first derivative
04:00 Gradient magnitude thresholding
06:30 The second derivative
08:00 Laplacian kernel
10:00 Why Laplacian sign matters
11:40 Zero crossings
14:00 Why derivatives amplify noise
16:30 C++ pipeline
19:00 Deterministic tests
21:30 Why this is not Canny
23:00 Next lesson
```

## Script outline

### 00:00 — What is an edge?

Show an image boundary and its 1-D intensity slice.

Define an edge as a rapid spatial intensity change, not simply "a white pixel in an edge image."

### 01:40 — First derivative

Use a step profile.

Show that the derivative becomes large around the transition.

Connect to Episode 06:

[
M = \sqrt{G_x^2 + G_y^2}
]

### 04:00 — Threshold magnitude

Write:

[
M \ge T \Rightarrow \text{edge candidate}
]

Explain that this makes an edge map but does not thin edges or reason about connectivity.

Demonstrate two thresholds and show the tradeoff.

### 06:30 — Second derivative

Show how differentiating the first derivative produces positive and negative lobes.

Explain that the sign transition is useful evidence of an edge location.

### 08:00 — Laplacian

Write:

[
\nabla^2 I = I_{xx}+I_{yy}
]

Show the 4-neighbor kernel:

[
\begin{bmatrix}
0&1&0\\
1&-4&1\\
0&1&0
end{bmatrix}
]

Then show that the self-test validates this kernel against `cv::Laplacian(..., ksize=1)`.

### 10:00 — Sign matters

Compare:

- signed Laplacian display
- absolute Laplacian display

Narration:

> Absolute value is useful for seeing response strength. It is destructive if the algorithm needs the zero crossing.

### 11:40 — Zero crossings

Explain the teaching rule:

- neighboring samples have opposite signs
- response difference exceeds a minimum threshold

Discuss why a threshold is needed to suppress tiny noisy oscillations.

### 14:00 — Noise

Show a nearly constant image with deterministic synthetic noise.

Compute Laplacian without smoothing.

Then Gaussian smooth and compute again.

The self-test verifies the smoothed Laplacian has lower mean absolute response.

### 16:30 — C++ implementation

Walk through:

- CLI parameters
- `CV_32F` conversion
- Gaussian smoothing
- Sobel magnitude
- computational thresholding
- signed Laplacian
- zero-crossing detector
- visualization helpers

Stress that the threshold is applied before display normalization.

### 19:00 — Tests

Run:

```bash
./build/cv9x_edge_derivatives --self-test
```

Explain:

- constant field
- linear ramp
- step edge
- threshold monotonicity
- explicit Laplacian equivalence
- noise suppression

Then:

```bash
ctest --test-dir build --output-on-failure
```

### 21:30 — Why this is not Canny

Show a checklist:

```text
Gaussian smoothing        yes
gradient calculation      yes
magnitude                  yes
non-maximum suppression   not yet
double threshold           not yet
hysteresis                 not yet
```

Set up Episode 08.

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_edge_derivatives

./build/cv9x_edge_derivatives \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/edge_derivative_results \
  --sigma 1.0 \
  --gradient-threshold 100 \
  --zero-crossing-threshold 15

./build/cv9x_edge_derivatives --self-test
ctest --test-dir build --output-on-failure
```

## Short idea

**Why Taking ABS Can Break Laplacian Edge Detection**

30–45 seconds:

1. show signed Laplacian
2. point to + / - sides
3. show zero crossing
4. apply absolute value
5. explain that the sign transition has disappeared

## YouTube description

In this lesson we connect image gradients to edge detection using first and second derivatives.

We build a Sobel-magnitude threshold edge map, compute the signed Laplacian, detect zero crossings, and demonstrate why Gaussian smoothing matters before differentiation.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/05_DerivativesEdgeDetection/00_DerivativesAndEdges

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- first vs second image derivatives
- gradient magnitude thresholding
- Laplacian math
- signed second derivatives
- zero crossings
- noise sensitivity
- deterministic edge tests

#ComputerVision #OpenCV #CPP #EdgeDetection #ImageProcessing #Linux

## Recording checklist

- [ ] animate 1-D step / derivatives
- [ ] show thresholded magnitude
- [ ] derive Laplacian kernel
- [ ] show signed vs absolute Laplacian
- [ ] show zero crossing
- [ ] demonstrate noise sensitivity
- [ ] show Gaussian smoothing effect
- [ ] walk through C++
- [ ] run self-test
- [ ] explain missing Canny stages
