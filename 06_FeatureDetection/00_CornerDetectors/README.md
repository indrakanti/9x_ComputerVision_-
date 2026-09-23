# Module 07 — Corner Detection: Harris, Shi-Tomasi & FAST

Edges tell us where image intensity changes strongly in one direction.

Corners are more useful for many vision tasks because they contain strong variation in **two directions** and are therefore easier to localize repeatedly.

This lesson unifies three important approaches:

- Harris corner response
- Shi-Tomasi minimum-eigenvalue criterion
- FAST high-speed corner detection

## Learning objectives

By the end of the lesson you should be able to:

- explain the second-moment / structure tensor
- classify flat regions, edges, and corners from its eigenvalues
- derive the Harris response
- explain the Shi-Tomasi criterion
- understand how Harris and Shi-Tomasi differ
- explain at a high level how FAST tests an intensity circle
- understand why FAST is computationally attractive
- separate detector response from point selection / non-maximum suppression
- validate detector behavior on synthetic flat, edge, and corner images

## Prerequisites

- image gradients
- Sobel derivatives
- local filtering / neighborhood aggregation
- Canny and edge concepts

## 1. Why edges are not enough

Imagine sliding a small image patch.

In a flat region, moving the patch slightly changes almost nothing.

Along an edge, motion parallel to the edge changes little, while motion perpendicular to it changes strongly.

At a corner, motion in nearly any direction changes the patch strongly.

That observation leads to the structure tensor.

## 2. Structure tensor

Using image gradients $I_x$ and $I_y$, define

$$
M =
\begin{bmatrix}
\sum I_x^2 & \sum I_x I_y \\
\sum I_x I_y & \sum I_y^2
\end{bmatrix}
$$

The sums are over a local window.

In the implementation, Sobel computes $I_x$ and $I_y$, then a local box filter aggregates:

- $I_x^2$
- $I_y^2$
- $I_xI_y$

## 3. Eigenvalue interpretation

Let the eigenvalues of $M$ be $\\lambda_1$ and $\\lambda_2$.

### Flat region

$$
\lambda_1 \approx 0, \quad \lambda_2 \approx 0
$$

There is little gradient energy in any direction.

### Edge

$$
\lambda_1 \gg \lambda_2
$$

There is one dominant gradient direction.

### Corner

$$
\lambda_1 \text{ and } \lambda_2 \text{ are both large}
$$

There is strong variation in two independent directions.

This flat/edge/corner interpretation is the conceptual center of the lesson.

## 4. Harris response

Harris avoids explicitly sorting eigenvalues.

It uses:

$$
R = \det(M) - k\operatorname{trace}(M)^2
$$

where typically:

$$
k \approx 0.04 \text{ to } 0.06
$$

Interpretation:

- (R \approx 0): flat
- $R < 0$: edge-like
- $R > 0$: corner-like

The CLI exposes `--harris-k`.

## 5. Shi-Tomasi

Shi-Tomasi uses a simpler score:

$$
R_{ST} = \min(\lambda_1, \lambda_2)
$$

A point is good only when the weaker eigen-direction is still strong.

This gives a very intuitive rule:

> A corner is only as good as its weaker direction.

The implementation computes the minimum eigenvalue directly from the 2x2 tensor components.

## 6. Selecting actual corner points

A response image is not yet a feature list.

We still need to:

1. threshold by response quality
2. perform local-maximum selection
3. enforce spatial separation
4. limit the number of returned corners

The canonical implementation exposes:

- `--harris-quality`
- `--shi-quality`
- `--max-corners`
- `--min-distance`

Quality is expressed as a fraction of each detector's maximum positive response.

## 7. FAST

FAST takes a different approach.

Rather than building a structure tensor, it examines a ring of pixels around a candidate point.

A point is considered corner-like when a sufficiently long contiguous arc on that circle is:

- brighter than the center by a threshold, or
- darker than the center by a threshold

OpenCV's FAST implementation uses a 16-pixel circle for `TYPE_9_16`.

This lesson calls `cv::FAST()` because the goal is to compare detector families after deriving Harris and Shi-Tomasi from first principles.

## 8. Harris vs Shi-Tomasi vs FAST

| Detector | Core idea | Response | Strength |
|---|---|---|---|
| Harris | determinant vs trace of structure tensor | continuous scalar | strong mathematical interpretation |
| Shi-Tomasi | minimum tensor eigenvalue | continuous scalar | intuitive "weakest direction" criterion |
| FAST | intensity circle test | discrete keypoint test / score | very fast detection |

FAST is not simply "a faster Harris." It is a different detector design.

## 9. Response vs descriptor

A detector answers:

> Where are interesting, repeatable points?

A descriptor answers:

> How do I represent the neighborhood around each point for matching?

This lesson covers detection only.

SIFT/ORB/BRIEF descriptors and matching come later.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_corners
```

## Run

```bash
./build/cv9x_corners \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/corner_results \
  --harris-k 0.04 \
  --harris-quality 0.01 \
  --shi-quality 0.01 \
  --max-corners 250 \
  --min-distance 8 \
  --fast-threshold 20
```

## Outputs

```text
01_harris_response.png
02_harris_corners.png
03_shi_tomasi_response.png
04_shi_tomasi_corners.png
05_fast_keypoints.png
```

## Deterministic validation

```bash
./build/cv9x_corners --self-test
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- flat tensor -> two near-zero eigenvalues
- edge tensor -> one strong and one weak eigenvalue
- corner tensor -> two strong eigenvalues
- Harris response is zero/negative/positive for flat/edge/corner idealizations
- constant image returns no corners
- a synthetic filled square produces Harris corners near all four geometric corners
- the same square produces Shi-Tomasi corners near all four corners
- an explicit synthetic FAST-9 circle pattern is detected at its center
- a pure straight edge does not create a positive Harris corner maximum
- a pure straight edge has near-zero Shi-Tomasi minimum eigenvalue

## Why the canonical lesson is different from the legacy demos

The older repository material contains useful separate demonstrations of:

- covariance / structure-tensor ideas
- eigenvalues
- Harris response
- region classification

The new lesson does not delete those examples.

Instead, it provides one reproducible end-to-end path with:

- explicit CLI inputs
- deterministic synthetic tests
- Harris and Shi-Tomasi derived from the same tensor
- point selection
- FAST comparison
- CMake / CTest integration
- recording-ready documentation

## Common failure cases

### Thresholding a normalized response image

Normalization is for visualization. Point selection should operate on the real response.

### Calling every high-response pixel a separate corner

A real corner creates a neighborhood of responses. Use local maxima and spacing.

### Confusing eigenvalue magnitude with eigenvector direction

Eigenvalues describe how much variation exists along principal directions. Eigenvectors describe those directions.

### Assuming FAST produces a Harris-like response map

FAST is based on a discrete circle test, not the Harris determinant/trace equation.

### Mixing detection and description

Corner detection does not by itself provide a robust matching descriptor.

## Engineering notes

### Detection output is an interface

A feature detector should define:

- coordinate convention
- response / score
- scale, if available
- orientation, if available
- maximum feature count
- suppression / spacing policy
- deterministic ordering

### Thresholds depend on image scale

Harris and Shi response values depend on:

- input intensity scale
- derivative scaling
- tensor window
- smoothing / aggregation
- image bit depth

This lesson therefore uses relative quality thresholds for response-map selection.

### Feature distribution matters

A detector that returns 500 points in one textured region may be less useful than one that distributes points across the image.

Later lessons can add grid-based or adaptive feature budgeting.

## Exercises

1. Replace the box aggregation with Gaussian-weighted tensor smoothing.
2. Compare Harris point selection with `cv::goodFeaturesToTrack(..., useHarrisDetector=true)`.
3. Compare manual Shi-Tomasi with `cv::goodFeaturesToTrack(..., useHarrisDetector=false)`.
4. Sweep Harris (k) and observe edge rejection vs corner response.
5. Sweep FAST threshold and plot keypoint count.
6. Add grid-based feature budgeting for more uniform spatial coverage.

## Next lesson

Episode 10 starts **SIFT scale space and Difference of Gaussians**, moving from single-scale corners to scale-invariant keypoints.

See [VIDEO.md](VIDEO.md).
