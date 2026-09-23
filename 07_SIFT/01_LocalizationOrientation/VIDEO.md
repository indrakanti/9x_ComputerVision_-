# Video: SIFT Part 2 — Subpixel Localization, Edge Rejection & Orientation

YouTube: TBD

Suggested title:

**SIFT Part 2 — Subpixel Localization, Edge Rejection & Orientation in C++**

Suggested thumbnail:

**RAW EXTREMUM -> KEYPOINT**

Target duration: **28–34 minutes**

## Learning objectives

Viewers should understand:

- why DoG extrema need refinement
- the 3-D Taylor approximation
- the offset equation $\Delta=-H^{-1}\nabla D$
- low-contrast rejection
- Hessian principal-curvature edge rejection
- scale refinement
- weighted orientation histograms
- why SIFT can duplicate one location with multiple orientations

## Chapters

```text
00:00 Raw extrema are not keypoints
02:00 3-D Taylor localization
06:00 Solve for subpixel/subscale offset
10:00 Iterative relocation
12:00 Contrast rejection
14:30 Why strong edges are bad features
17:00 Hessian curvature-ratio test
20:00 Refined scale
22:00 Orientation histogram
26:00 Gaussian weighting + circular smoothing
29:00 Multiple orientations
31:00 C++ implementation + deterministic tests
33:00 Next: SIFT descriptor
```

## Script outline

### 00:00 — Raw extrema are not keypoints

Show the output from Part 1.

Zoom into one raw DoG extremum.

Explain:

> This coordinate only says which sampled pixel and sampled scale won the 26-neighbor test. The true extremum can live between samples.

### 02:00 — Taylor model

Write:

$$
D(\mathbf{x}+\Delta)
\approx
D+
\nabla D^T\Delta+
\frac12\Delta^TH\Delta
$$

Then set derivative to zero:

$$
\Delta=-H^{-1}\nabla D
$$

Draw the three axes:

```text
x
y
scale
```

### 06:00 — Fractional offset

Use the deterministic synthetic quadratic from the self-test.

Show the known true maximum:

```text
x = 4.25
y = 3.75
scale = 1.20
```

Start from integer:

```text
(4, 4, 1)
```

and show the recovered offset.

### 10:00 — Iterative relocation

Explain the half-sample rule.

If the Taylor estimate says the peak is closer to a neighboring sample, move there and recompute.

Show failure exits:

- boundary
- singular Hessian
- non-convergence

### 12:00 — Contrast

Write:

$$
D(\hat{\mathbf{x}})
\approx
D + \frac12\nabla D^T\Delta
$$

Explain that weak extrema are unstable.

Stress that the lesson threshold is in raw DoG units.

### 14:30 — Why edges are rejected

Show:

- isotropic blob
- long ridge

A ridge may have strong response but poor localization along its long direction.

### 17:00 — Hessian curvature test

Write:

$$
H_{xy} =
\begin{bmatrix}
D_{xx}&D_{xy}\\
D_{xy}&D_{yy}
\end{bmatrix}
$$

Then:

$$
\frac{Tr^2}{Det}
<
\frac{(r+1)^2}{r}
$$

Explain why the determinant must also be positive.

### 20:00 — Refined scale

Write:

$$
\sigma=\sigma_0k^{l+\Delta_s}
$$

Then map to original-image scale by multiplying by the octave factor.

### 22:00 — Orientation histogram

Show local gradients around the keypoint.

Use:

$$
m=\sqrt{G_x^2+G_y^2}
$$

$$
\theta=\operatorname{atan2}(G_y,G_x)
$$

Show the 36-bin circular histogram.

### 26:00 — Weighting and smoothing

Explain:

```text
near keypoint -> stronger weight
farther away  -> weaker weight
```

Then show circular histogram smoothing and parabolic peak interpolation.

### 29:00 — Multiple orientations

Create a conceptual two-direction patch.

Explain the 80% rule:

> A strong secondary orientation becomes a second oriented keypoint at the same position and scale.

This is not a duplicate bug. It is how SIFT handles ambiguous local orientation.

### 31:00 — Code and tests

Walk through:

- `derivativeModel()`
- `solveOffset()`
- `refineCandidate()`
- `passesEdgeResponse()`
- `buildOrientationHistogram()`
- `orientationPeaks()`

Run:

```bash
./build/cv9x_sift_localization_orientation --self-test
ctest --test-dir build --output-on-failure
```

Show the three output images.

### 33:00 — Next lesson

Preview:

> Part 3 rotates a local frame around each oriented keypoint and builds the 128-number SIFT descriptor used for matching.

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_sift_localization_orientation

./build/cv9x_sift_localization_orientation \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/sift_part2 \
  --candidate-threshold 1 \
  --contrast-threshold 2 \
  --edge-ratio 10 \
  --orientation-bins 36 \
  --orientation-peak-ratio 0.8

./build/cv9x_sift_localization_orientation --self-test
ctest --test-dir build --output-on-failure
```

## Short idea

**Why SIFT Throws Away Strong Edges**

30–45 seconds:

1. show a strong ridge
2. say "high response does not mean good feature"
3. slide the point along the ridge
4. show localization ambiguity
5. reveal the Hessian curvature-ratio test

## YouTube description

SIFT Part 2 turns raw Difference-of-Gaussian extrema into stable oriented keypoints.

We implement subpixel/subscale Taylor refinement, low-contrast rejection, Hessian edge-response filtering, and the weighted orientation histogram from first principles in C++.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/07_SIFT/01_LocalizationOrientation

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- 3-D SIFT keypoint localization
- Taylor expansion and Hessian solve
- contrast rejection
- principal-curvature edge rejection
- orientation histograms
- multiple SIFT orientations
- deterministic synthetic tests

#ComputerVision #OpenCV #CPP #SIFT #FeatureDetection #Linux

## Recording checklist

- [ ] show raw extrema from Part 1
- [ ] derive Taylor offset
- [ ] animate iterative relocation
- [ ] explain interpolated contrast
- [ ] compare blob vs ridge
- [ ] derive Hessian curvature ratio
- [ ] explain refined sigma
- [ ] build orientation histogram
- [ ] explain Gaussian weighting
- [ ] explain secondary orientation peaks
- [ ] run deterministic tests
- [ ] preview 128-element descriptor
