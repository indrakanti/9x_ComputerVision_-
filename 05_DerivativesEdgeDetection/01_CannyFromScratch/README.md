# Module 06 — Canny Edge Detection from Scratch

This lesson implements the complete Canny pipeline stage by stage in C++.

The implementation does **not** call `cv::Canny()` to produce its result. OpenCV Canny is used only at the end as a reference image for comparison.

## Learning objectives

By the end of this lesson you should be able to explain and implement:

1. Gaussian smoothing
2. Sobel (G_x) and (G_y)
3. gradient magnitude and direction
4. direction quantization
5. non-maximum suppression
6. double thresholding
7. hysteresis edge tracking

You should also understand why Canny produces a much more useful edge map than simply thresholding gradient magnitude.

## Prerequisites

- convolution
- Gaussian filtering
- Sobel / Scharr gradients
- first and second derivatives
- basic edge thresholding

## The full pipeline

```text
image
  |
  v
Gaussian smoothing
  |
  v
Sobel Gx / Gy
  |
  v
magnitude + direction
  |
  v
non-maximum suppression
  |
  v
double threshold
  |
  v
8-connected hysteresis
  |
  v
final binary edges
```

## 1. Gaussian smoothing

Differentiation amplifies high-frequency noise.

The first stage therefore smooths the image:

$$
I_s = G_\sigma * I
$$

The CLI exposes `--sigma`.

A value of zero disables smoothing for experimentation.

## 2. Sobel gradients

We compute:

$$
G_x = S_x * I_s
$$

$$
G_y = S_y * I_s
$$

and preserve them in `CV_32F`.

Magnitude:

$$
M = \sqrt{G_x^2 + G_y^2}
$$

Direction:

$$
\theta = \operatorname{atan2}(G_y,G_x)
$$

OpenCV `cv::phase()` returns the direction in degrees.

## 3. Why magnitude thresholding alone is not enough

A strong edge often creates a thick ridge of high gradient magnitude.

If every high-magnitude pixel becomes an edge, the result is wide and poorly localized.

Canny solves this using **non-maximum suppression**.

## 4. Direction quantization

The exact gradient direction is reduced to four edge-normal directions:

```text
0 degrees
45 degrees
90 degrees
135 degrees
```

Because a gradient axis is periodic every 180 degrees:

```text
0   == 180
45  == 225
90  == 270
135 == 315
```

The implementation uses these ranges:

| Angle modulo 180 | Quantized direction |
|---|---:|
| [0, 22.5) or [157.5, 180) | 0 |
| [22.5, 67.5) | 45 |
| [67.5, 112.5) | 90 |
| [112.5, 157.5) | 135 |

## 5. Non-maximum suppression

For each pixel:

1. read its gradient magnitude
2. quantize its direction
3. compare it to the two neighboring magnitudes along that direction
4. keep the pixel only if it is a local maximum

Conceptually:

```text
before  <-- center --> after
```

If the center is not at least as strong as both comparison neighbors, it is suppressed to zero.

This thins broad gradient ridges into edge candidates.

## 6. Double threshold

One threshold is often too brittle.

Canny uses two:

- **high threshold** -> strong edge
- **low threshold** -> possible weak edge
- below low -> reject

This lesson stores the intermediate labels as:

```text
0   = rejected
75  = weak
255 = strong
```

The saved visualization maps weak edges to gray 128 and strong edges to white.

## 7. Hysteresis

A weak pixel should survive only when it is connected to a strong edge.

The implementation:

1. inserts every strong pixel into a queue
2. performs an 8-connected graph traversal
3. promotes connected weak pixels into final edges
4. rejects isolated weak pixels

This is the connectivity reasoning missing from simple thresholding.

## 8. Why two thresholds help

Imagine a real edge that becomes weaker because of:

- illumination variation
- texture
- blur
- partial occlusion
- sensor noise

A single high threshold can break that edge into pieces.

A low threshold alone keeps too much noise.

Hysteresis combines both ideas:

> Keep weak evidence only when strong evidence supports it through connectivity.

## 9. OpenCV reference comparison

The program also runs:

```cpp
cv::Canny(..., L2gradient=true)
```

on the equivalently smoothed 8-bit image.

It writes:

```text
07_opencv_canny_reference.png
08_reference_difference.png
```

and prints a binary intersection-over-union value.

Do **not** expect pixel-perfect equality.

OpenCV uses implementation details and optimized internal paths that can differ in:

- border treatment
- rounding
- NMS tie behavior
- gradient calculations
- threshold handling

The reference is for engineering comparison, not a unit-test oracle.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_canny
```

## Run

```bash
./build/cv9x_canny \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/canny_results \
  --sigma 1.0 \
  --low-threshold 50 \
  --high-threshold 100 \
  --border reflect101
```

## Outputs

```text
01_smoothed.png
02_gradient_magnitude.png
03_gradient_direction.png
04_non_maximum_suppression.png
05_double_threshold.png
06_hysteresis_edges.png
07_opencv_canny_reference.png
08_reference_difference.png
```

The numbered filenames make the pipeline easy to inspect in order.

## Deterministic self-test

```bash
./build/cv9x_canny --self-test
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- angle quantization including 180-degree wraparound
- NMS keeps a local maximum
- NMS suppresses its weaker neighbors
- double threshold creates rejected / weak / strong classes
- hysteresis follows an 8-connected weak chain from a strong pixel
- hysteresis rejects an isolated weak pixel
- constant images produce no final edges
- a synthetic vertical step produces strong and final edges
- the stored direction field matches the phase calculation

## Common failure cases

### Comparing in the wrong direction during NMS

NMS compares magnitude along the **gradient direction**, which is normal to the edge.

### Converting directions to integers too early

Preserve the floating-point direction field, then quantize deliberately.

### Keeping every weak edge

Weak edges survive only if connected to a strong edge.

### Using 4-connectivity accidentally

Classic Canny hysteresis typically considers 8-connected neighbors.

### Swapping low and high thresholds

The implementation validates:

```text
0 <= low <= high
```

### Thresholding the display-normalized magnitude

Threshold the actual computational magnitude. Display normalization is only for visualization.

## Engineering notes

### Intermediate stages are interfaces

A production implementation should treat these as named data products:

- smoothed intensity
- signed gradients
- magnitude
- direction
- suppressed magnitude
- threshold classification
- final edge map

That makes the pipeline testable and diagnosable.

### Hysteresis is graph traversal

Once double thresholding is complete, the problem is no longer just image arithmetic. It becomes connectivity over an 8-neighbor graph.

### Thresholds are part of the algorithm contract

Document their units, input bit depth, Sobel scaling, and preprocessing. A threshold copied from another pipeline may be meaningless if any of those change.

### Optimized Canny will look different internally

A production library may fuse stages, vectorize loops, quantize angles differently, or avoid storing intermediate images. The first-principles version is deliberately explicit so its behavior can be understood and tested.

## Exercises

1. Replace four-bin angle quantization with interpolation between neighboring gradient samples.
2. Compare 4-connected and 8-connected hysteresis.
3. Sweep low/high thresholds and record edge continuity.
4. Inject deterministic noise and compare results at several sigma values.
5. Add a `--no-reference` option to skip the OpenCV comparison.
6. Benchmark the explicit implementation against `cv::Canny()`.

## Next lesson

Episode 09 moves into **corner and feature detection**, beginning with Harris and Shi-Tomasi before FAST.

See [VIDEO.md](VIDEO.md).
