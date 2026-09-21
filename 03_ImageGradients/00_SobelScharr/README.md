# Module 05 — Sobel & Scharr Image Gradients

This is the canonical gradient lesson for **9x Computer Vision — From First Principles to Production C++ on Linux**.

The goal is not simply to call `cv::Sobel()`. The lesson explains what an image derivative means, why derivative responses are signed, how (G_x) and (G_y) become magnitude and direction, and why Scharr exists alongside Sobel.

## Learning objectives

By the end of the lesson you should be able to:

- explain a discrete image derivative as a local intensity-change measurement
- derive the 3x3 Sobel kernels from smoothing and differentiation components
- explain the sign of (G_x) and (G_y)
- compute gradient magnitude and direction
- distinguish algorithm data from visualization data
- explain why Scharr improves rotational accuracy for a 3x3 derivative
- understand how border policy changes derivative values near the image boundary
- validate OpenCV derivative operators against explicit kernels

## Prerequisites

Complete these first:

1. point operations
2. convolution
3. box and Gaussian filtering
4. image pyramids

This lesson is where filtering becomes **spatial differentiation**.

## 1. From intensity to derivative

For a continuous image (I(x,y)), the gradient is

[

abla I =
egin{bmatrix}
rac{partial I}{partial x} \
rac{partial I}{partial y}
end{bmatrix}
=
egin{bmatrix}
G_x \
G_y
end{bmatrix}
]

A digital image is sampled, so we approximate the derivatives with local finite-difference kernels.

A simple horizontal derivative is

[
[-1 quad 0 quad 1]
]

It asks:

> Is the neighborhood becoming brighter or darker as we move in x?

That sign matters.

- positive (G_x): intensity rises from left to right
- negative (G_x): intensity falls from left to right

Taking an absolute value too early destroys that polarity.

## 2. Sobel kernels

The 3x3 Sobel x-derivative is

[
G_x =
egin{bmatrix}
-1 & 0 & 1 \
-2 & 0 & 2 \
-1 & 0 & 1
end{bmatrix}
]

and the y-derivative is

[
G_y =
egin{bmatrix}
-1 & -2 & -1 \
0 & 0 & 0 \
1 & 2 & 1
end{bmatrix}
]

Sobel can be understood as combining:

- differentiation in one direction: ([-1,0,1])
- smoothing in the perpendicular direction: ([1,2,1]^T)

That is why Sobel is more useful than a raw one-dimensional difference in noisy images.

## 3. Gradient magnitude

Once we have (G_x) and (G_y),

[
M = sqrt{G_x^2 + G_y^2}
]

Magnitude answers:

> How strong is the local intensity change?

Magnitude is non-negative, but (G_x) and (G_y) themselves must remain signed.

## 4. Gradient direction

The orientation is

[
	heta = operatorname{atan2}(G_y, G_x)
]

The implementation uses OpenCV's `cv::phase(..., angleInDegrees=true)`, producing directions in degrees.

Examples for ideal ramps:

| Image ramp | Expected direction |
|---|---:|
| intensity increases left -> right | 0 deg |
| intensity increases top -> bottom | 90 deg |
| equal increase in x and y | 45 deg |

Gradient direction later matters in Canny, HOG, SIFT, feature descriptors, and many classical vision algorithms.

## 5. Why signed storage matters

A derivative is naturally signed.

Suppose the image contains a bright vertical stripe on a dark background.

At the left edge:

[
G_x > 0
]

At the right edge:

[
G_x < 0
]

If you immediately convert (G_x) to `CV_8U`, negative values cannot be represented correctly.

This lesson therefore computes in:

```text
CV_32F
```

and converts only when producing visualization PNGs.

For signed display:

- zero derivative maps near gray 128
- negative derivative maps darker
- positive derivative maps brighter

The display image is not the algorithm data.

## 6. Sobel vs Scharr

For a 3x3 derivative, Scharr uses coefficients chosen for better rotational symmetry.

Scharr x kernel:

[
egin{bmatrix}
-3 & 0 & 3 \
-10 & 0 & 10 \
-3 & 0 & 3
end{bmatrix}
]

Scharr y is its transpose.

Sobel is excellent for teaching and widely used. Scharr is useful when you specifically want a 3x3 derivative with improved isotropy.

Do not interpret the raw Sobel and Scharr magnitudes as directly comparable without considering their different kernel gains.

## 7. Border handling

A derivative near the image boundary needs pixels outside the image.

This lesson exposes:

- `reflect101`
- `replicate`

Border policy is part of the numerical definition of the operation. It can change derivative values around the perimeter and should be explicit in reproducible pipelines.

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_gradients
```

## Run

```bash
./build/cv9x_gradients \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/gradient_results \
  --method both \
  --border reflect101
```

Supported methods:

```text
sobel
scharr
both
```

## Outputs

For Sobel:

```text
sobel_gx_signed.png
sobel_gy_signed.png
sobel_magnitude.png
sobel_direction.png
```

For Scharr:

```text
scharr_gx_signed.png
scharr_gy_signed.png
scharr_magnitude.png
scharr_direction.png
```

The signed derivative PNGs are visualizations centered around gray. The internal derivatives remain `CV_32F`.

## Deterministic validation

Run:

```bash
./build/cv9x_gradients --self-test
```

or all lesson tests:

```bash
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- constant images produce zero gradient
- an x-ramp produces positive (G_x), near-zero (G_y), and 0-degree direction
- a y-ramp produces near-zero (G_x), positive (G_y), and 90-degree direction
- a diagonal ramp produces a 45-degree direction
- explicit Sobel kernels match `cv::Sobel()`
- explicit Scharr kernels match `cv::Scharr()`
- a bright stripe produces both positive and negative x-derivative responses

## Common failure cases

### 1. Storing derivatives in unsigned 8-bit images

Negative gradients disappear or saturate.

**Fix:** use a signed or floating-point derivative type such as `CV_32F`.

### 2. Calling magnitude "the Sobel image"

Magnitude is derived from two directional derivative fields. Preserve (G_x) and (G_y) when direction or polarity matters.

### 3. Using `convertScaleAbs()` inside the algorithm

It is useful for display, but absolute value destroys the derivative sign.

### 4. Ignoring border mode

Two implementations can disagree only near the edges because they use different border assumptions.

### 5. Comparing Sobel and Scharr raw magnitude directly

Their coefficient scales differ. Normalize appropriately before making a quantitative comparison.

## Engineering notes

### Keep computation and visualization separate

Production pipelines frequently need signed gradients for later stages. Visualization should be a terminal/debug transformation, not part of the computational contract.

### Derivatives amplify noise

Differentiation is a high-frequency operation. This is why smoothing and derivative design are connected, and why Canny starts with Gaussian smoothing.

### Direction is circular

0 degrees and 360 degrees represent the same direction. Angle comparisons must account for wraparound.

### Precision matters downstream

If a later stage needs non-maximum suppression or orientation histograms, preserve enough precision until that stage is complete.

## Exercises

1. Add Prewitt x/y kernels and compare them to Sobel on the same image.
2. Add an option for Sobel kernel sizes 3, 5, and 7. Measure how the response and runtime change.
3. Normalize Sobel and Scharr gains and compare their response to edges rotated through several angles.
4. Add Gaussian pre-smoothing and measure gradient stability after injecting synthetic noise.
5. Create an orientation-color visualization rather than an 8-bit grayscale direction image.

## Next lesson

Episode 07 moves from gradient fields to **image derivatives and edge detection**, preparing for a complete Canny implementation with non-maximum suppression and hysteresis.

See [VIDEO.md](VIDEO.md) for the recording plan.
