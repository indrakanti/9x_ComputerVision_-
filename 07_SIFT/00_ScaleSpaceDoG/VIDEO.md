# Video: SIFT Part 1 — Scale Space & Difference of Gaussians from First Principles

YouTube: TBD

Suggested title:

**SIFT Part 1 — Scale Space & Difference of Gaussians from First Principles**

Suggested thumbnail:

**FIND FEATURES AT ANY SCALE**

Target duration: **26–32 minutes**

## Learning objectives

Viewers should understand:

- why single-scale corners are not enough
- Gaussian scale space
- octaves and intervals
- $k=2^{1/s}$
- absolute vs incremental sigma
- why SIFT uses $s+3$ Gaussian images
- Difference of Gaussians
- 26-neighbor extrema detection
- why raw extrema are not yet final keypoints

## Visual hook

Start with the same object shown at three apparent sizes.

Overlay a fixed-size corner detector window.

Ask:

> If the image feature becomes twice as large, why should the same fixed-size neighborhood still detect it?

Then show a Gaussian scale stack and highlight the layer where the structure becomes most distinctive.

## Chapters

```text
00:00 The scale problem
02:00 What is Gaussian scale space?
05:00 Octaves and intervals
08:00 Deriving k = 2^(1/s)
10:00 Absolute vs incremental sigma
14:00 Building a Gaussian octave
17:00 Octave transitions
19:30 Difference of Gaussians
22:30 DoG vs Laplacian intuition
25:00 26-neighbor extrema
28:00 C++ implementation + tests
31:00 What Part 1 does NOT do
```

## Script outline

### 00:00 — The scale problem

Show the same corner/blob at multiple sizes.

Connect back to Harris/FAST:

> Those detectors worked at the image scale we gave them. SIFT asks a different question: at what scale is this structure most stable?

### 02:00 — Gaussian scale space

Write:

$$
L(x,y,\sigma)=G(x,y,\sigma)*I(x,y)
$$

Explain that increasing sigma progressively removes fine structure.

Use a visual stack of the same image at several sigma values.

### 05:00 — Octaves and intervals

Explain:

- octave = factor-of-two scale range
- interval = discrete scale step inside an octave

For $s$ intervals:

$$
k=2^{1/s}
$$

### 08:00 — Why k works

Show:

$$
k^s = 2
$$

For $s=3$:

$$
k=2^{1/3}
$$

Then list:

```text
sigma0
sigma0*k
sigma0*k^2
sigma0*k^3 = 2*sigma0
...
```

### 10:00 — Incremental sigma

This is the implementation detail worth emphasizing.

Write:

$$
\sigma_{inc}
=
\sqrt{
\sigma_{target}^2-
\sigma_{prev}^2
}
$$

Narration:

> Gaussian variances add. If the image is already blurred, we only add the missing variance.

Show how using the absolute target sigma repeatedly over-blurs the image.

### 14:00 — Gaussian octave

Show six images for $s=3$.

Explain why SIFT uses $s+3$, not just (s).

### 17:00 — Next octave

Highlight Gaussian level $s$.

Show direct factor-of-two downsampling.

Explain why we do not call `pyrDown()` here: it would add extra blur and complicate the sigma accounting.

### 19:30 — DoG

Write:

$$
D_i=L_{i+1}-L_i
$$

Show signed DoG visualization.

Explain dark/gray/bright mapping is only for display.

### 22:30 — DoG and LoG

Show the conceptual approximation:

$$
G(k\sigma)-G(\sigma)
\propto
\sigma^2\nabla^2G
$$

Connect back to the Laplacian lesson.

Keep the derivation intuitive rather than turning this into a calculus lecture.

### 25:00 — 26-neighbor extrema

Draw three (3\times3) DoG patches:

```text
previous scale
current scale
next scale
```

Put the candidate at the center of the current scale.

Explain:

> It must beat all 26 neighbors, or be lower than all 26.

That makes the search three-dimensional: x, y, and scale.

### 28:00 — C++ implementation

Walk through:

- `scaleStep()`
- `absoluteSigmas()`
- `incrementalSigmas()`
- `buildGaussianPyramid()`
- `buildDoGPyramid()`
- `isScaleSpaceExtremum()`

Run:

```bash
./build/cv9x_sift_scale_space --self-test
```

Point out tests for sigma bookkeeping, constant-image DoG, octave dimensions, adjacent subtraction, and explicit positive/negative extrema.

### 31:00 — What Part 1 does not do

Show:

```text
scale-space candidate
   |
   v
localization / rejection     Part 2
   |
   v
orientation                  Part 2
   |
   v
descriptor                   Part 3
```

This prevents viewers from confusing a DoG extremum with a finished SIFT feature.

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_sift_scale_space

./build/cv9x_sift_scale_space \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/sift_scale_space \
  --octaves 4 \
  --intervals 3 \
  --sigma 1.6 \
  --contrast-threshold 5

./build/cv9x_sift_scale_space --self-test
ctest --test-dir build --output-on-failure
```

## Visuals to show

- Gaussian octave montage
- DoG octave montage
- signed DoG response
- octave size reduction
- 3x3x3 extrema neighborhood
- raw extrema overlay

## Short idea

**Why SIFT Uses 6 Gaussian Images for 3 Scale Intervals**

30–45 seconds:

1. say "3 intervals does NOT mean 3 Gaussian images"
2. show (s+3=6)
3. subtract neighbors -> (s+2=5) DoGs
4. explain extrema need previous/current/next scale
5. finish with "extra images create valid scale neighborhoods"

## YouTube description

In SIFT Part 1 we build scale space from first principles in C++ on Linux.

We derive the octave scale multiplier, distinguish absolute and incremental Gaussian sigma, construct Gaussian and Difference-of-Gaussian pyramids, and implement the 26-neighbor scale-space extrema test.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/07_SIFT/00_ScaleSpaceDoG

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- Gaussian scale space
- SIFT octaves and intervals
- sigma bookkeeping
- Difference of Gaussians
- DoG / Laplacian intuition
- 26-neighbor extrema
- deterministic scale-space tests

#ComputerVision #OpenCV #CPP #SIFT #FeatureDetection #Linux

## Recording checklist

- [ ] demonstrate the scale problem
- [ ] derive k
- [ ] explain s+3 Gaussian levels
- [ ] derive incremental sigma
- [ ] show Gaussian octave montage
- [ ] explain octave downsampling
- [ ] show signed DoG
- [ ] draw 26-neighbor test
- [ ] walk through core C++
- [ ] run self-test
- [ ] explain why extrema are not final keypoints
- [ ] preview Part 2 localization/orientation
