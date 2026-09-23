# Video: Corner Detection in C++ — Harris, Shi-Tomasi & FAST Explained

YouTube: TBD

Suggested title:

**Corner Detection in C++ — Harris, Shi-Tomasi & FAST Explained**

Suggested thumbnail:

**EDGE OR CORNER?**

Target duration: **24–30 minutes**

## Learning objectives

Viewers should understand:

- why corners are more localizable than edges
- the structure tensor
- flat / edge / corner eigenvalue patterns
- Harris response
- Shi-Tomasi minimum eigenvalue
- FAST intensity-circle intuition
- response thresholding and non-maximum selection
- detector vs descriptor

## Visual hook

Show three patches:

1. flat wall
2. straight edge
3. checkerboard corner

Ask:

> If I shift this patch a few pixels, which one can I relocate most precisely?

Then animate patch motion:

- flat: almost no change
- edge: ambiguous along one axis
- corner: strong change in every direction

## Chapters

```text
00:00 Why corners matter
01:50 Flat vs edge vs corner
04:00 The structure tensor
07:00 Eigenvalues as geometry
10:00 Harris response
13:00 Shi-Tomasi
15:30 Selecting local maxima
18:00 FAST detector intuition
21:00 C++ implementation
24:30 Deterministic synthetic tests
27:00 Harris vs Shi-Tomasi vs FAST
29:00 Next: SIFT scale space
```

## Script outline

### 00:00 — Why corners matter

Connect to tracking, matching, localization, SLAM, panorama stitching, and calibration.

Explain that an edge constrains position well in only one direction.

### 01:50 — Flat / edge / corner

Use the classic shifted-window intuition.

Draw:

```text
flat:    little change
edge:    change in one direction
corner:  change in two directions
```

### 04:00 — Structure tensor

Build from the gradients already taught:

$$
M =
\begin{bmatrix}
\sum I_x^2 & \sum I_xI_y \\
\sum I_xI_y & \sum I_y^2
\end{bmatrix}
$$

Explain the matrix as local directional gradient energy.

### 07:00 — Eigenvalues

Show:

```text
flat    lambda1 small, lambda2 small
edge    one large, one small
corner  both large
```

Use the synthetic tensor self-tests in the code to make this concrete.

### 10:00 — Harris

Write:

$$
R = \det(M) - k\operatorname{trace}(M)^2
$$

Explain:

- corner -> positive
- edge -> negative
- flat -> near zero

### 13:00 — Shi-Tomasi

Write:

$$
R = \min(\lambda_1,\lambda_2)
$$

Narration:

> A point is only as corner-like as its weaker principal direction.

Compare response maps.

### 15:30 — Point selection

Explain why a response image is not yet a keypoint list.

Show:

1. quality threshold
2. local maxima
3. minimum spacing
4. maximum feature count

### 18:00 — FAST

Draw the 16-pixel circle.

Explain the core idea without turning the lesson into a full FAST derivation:

- compare circle pixels with center
- require a contiguous bright/dark arc
- use threshold
- optionally suppress neighbors

Explain why it is attractive for real-time systems.

### 21:00 — C++ implementation

Walk through:

- `computeStructureTensor()`
- `harrisResponse()`
- `shiTomasiResponse()`
- `selectLocalMaxima()`
- `cv::FAST()`

Emphasize shared tensor math between Harris and Shi.

### 24:30 — Tests

Run:

```bash
./build/cv9x_corners --self-test
```

Show:

- ideal flat tensor
- ideal edge tensor
- ideal corner tensor
- constant image
- filled square
- straight edge

Explain why synthetic images are excellent algorithm tests: the expected geometry is known.

Then:

```bash
ctest --test-dir build --output-on-failure
```

### 27:00 — Compare detectors

Use a table:

```text
Harris       tensor + determinant/trace
Shi-Tomasi  tensor + min eigenvalue
FAST         intensity circle test
```

Discuss quality vs speed without declaring one detector universally best.

### 29:00 — Next lesson

Introduce scale:

> A corner found at one image resolution may disappear when the camera moves closer or farther away. SIFT addresses that with scale space.

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_corners

./build/cv9x_corners \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/corner_results \
  --harris-k 0.04 \
  --harris-quality 0.01 \
  --shi-quality 0.01 \
  --max-corners 250 \
  --min-distance 8 \
  --fast-threshold 20

./build/cv9x_corners --self-test
ctest --test-dir build --output-on-failure
```

## Short idea

**Why an Edge Is a Bad Feature but a Corner Is Good**

30–45 seconds:

1. show straight edge
2. slide window along edge
3. say "same appearance — ambiguous location"
4. show corner
5. move in x/y
6. say "appearance changes both ways — easier to localize"

## YouTube description

In this lesson we move from edge detection to feature detection and implement Harris and Shi-Tomasi corners from the structure tensor, then compare them with FAST.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/06_FeatureDetection/00_CornerDetectors

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- structure tensor
- flat / edge / corner eigenvalues
- Harris response
- Shi-Tomasi minimum eigenvalue
- local maximum corner selection
- FAST detector intuition
- synthetic feature-detector tests

#ComputerVision #OpenCV #CPP #HarrisCorner #FAST #FeatureDetection #Linux

## Recording checklist

- [ ] show flat/edge/corner patch motion
- [ ] derive structure tensor
- [ ] explain eigenvalues visually
- [ ] derive Harris response
- [ ] explain Shi-Tomasi
- [ ] show local-max / spacing selection
- [ ] draw FAST circle
- [ ] walk through C++
- [ ] run synthetic tests
- [ ] compare three detectors
- [ ] preview SIFT scale space
