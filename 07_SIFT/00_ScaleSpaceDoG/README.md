# Module 08 — SIFT Part 1: Scale Space & Difference of Gaussians

This is the first of three SIFT lessons.

The purpose of Part 1 is to answer one question:

> How can a feature detector find the same physical structure when that structure appears at a different image scale?

The answer is **scale space**.

This lesson builds the Gaussian pyramid, constructs Difference-of-Gaussian (DoG) images, and detects raw 3-D extrema across $x$, $y$, and scale.

It intentionally stops before SIFT keypoint localization, edge-response rejection, orientation assignment, and descriptor construction. Those are later lessons.

## Learning objectives

By the end of this lesson you should be able to:

- explain why single-scale feature detection is not scale invariant
- define Gaussian scale space
- explain octaves and intervals
- derive the scale multiplier $k$
- distinguish absolute sigma from incremental blur
- build a Gaussian octave correctly
- build a DoG octave by subtracting adjacent Gaussian levels
- explain why DoG approximates the scale-normalized Laplacian
- explain the 26-neighbor scale-space extrema test
- understand why a raw DoG extremum is not yet a final SIFT keypoint

## Prerequisites

- Gaussian filtering
- image pyramids
- Laplacian / second derivatives
- corner and feature detection

## 1. The scale problem

A fixed-size detector observes a fixed-size neighborhood.

If an object moves closer to the camera, the same physical structure occupies more pixels.

If it moves farther away, it occupies fewer pixels.

A detector operating at only one scale can therefore miss the same feature after a scale change.

SIFT solves this by searching for stable structure across a family of progressively blurred and resized images.

## 2. Gaussian scale space

For an image $I(x,y)$, Gaussian scale space is

$$
L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)
$$

where

$$
G(x,y,\sigma) =
\frac{1}{2\pi\sigma^2}
\exp\left(
-\frac{x^2+y^2}{2\sigma^2}
\right)
$$

Increasing $\\sigma$ removes progressively finer image structure.

## 3. Octaves and intervals

One **octave** spans a doubling of scale.

If an octave contains $s$ intervals, define

$$
k = 2^{1/s}
$$

and Gaussian levels

$$
\sigma_i = \sigma_0 k^i
$$

With the common SIFT choice $s=3$:

$$
k = 2^{1/3}
$$

After three intervals:

$$
k^3 = 2
$$

The self-test verifies this relationship numerically.

## 4. Why SIFT needs extra Gaussian images

For $s$ scale intervals, the lesson constructs:

$$
s+3
$$

Gaussian images per octave.

That produces:

$$
s+2
$$

DoG images.

The extra levels are necessary because scale-space extrema need a previous, current, and next DoG layer.

With $s=3$:

```text
Gaussian images: 6
DoG images:      5
Extrema-tested DoG layers: the interior layers
```

## 5. Absolute sigma vs incremental sigma

This is an important implementation detail.

Suppose one Gaussian image is already blurred to $\\sigma_{prev}$, and we want the next level to have total blur $\\sigma_{target}$.

We must not blur it again with $\\sigma_{target}$.

Gaussian variances add:

$$
\sigma_{target}^2 =
\sigma_{prev}^2 +
\sigma_{increment}^2
$$

Therefore:

$$
\sigma_{increment}
=
\sqrt{
\sigma_{target}^2 -
\sigma_{prev}^2
}
$$

The implementation keeps both concepts explicit.

## 6. Octave transition

Within one octave, scale increases geometrically.

The Gaussian image at interval index $s$ has twice the base sigma:

$$
\sigma_s = \sigma_0 k^s = 2\sigma_0
$$

That image is downsampled by two to become the next octave's base.

After halving image resolution, its blur relative to the new pixel spacing is again consistent with \(\sigma_0\). The next octave therefore uses that downsampled image **directly as layer 0**; it does not apply \(\sigma_0\) a second time.

The implementation uses direct factor-of-two subsampling rather than `pyrDown()` so that octave transition does not silently add another Gaussian blur.

## 7. Difference of Gaussians

For adjacent Gaussian levels:

$$
D_i = L_{i+1} - L_i
$$

The DoG is signed.

Positive and negative responses carry different information, so the internal images remain `CV_32F`.

The saved PNGs are only visualizations:

- negative -> dark
- zero -> gray
- positive -> bright

## 8. Why DoG is useful

The Difference of Gaussians efficiently approximates a scale-normalized Laplacian response.

Conceptually:

$$
G(x,y,k\sigma) - G(x,y,\sigma)
\approx
C\sigma^2\nabla^2 G
$$

for a scale-dependent constant $C$.

This gives SIFT a practical way to search for blob-like structures across scale without explicitly evaluating a full Laplacian-of-Gaussian filter at every scale.

## 9. Scale-space extrema

A DoG sample is considered a raw candidate when it is strictly greater than all neighbors or strictly less than all neighbors in a (3\times3\times3) neighborhood.

That means comparing against:

- 9 samples in the previous DoG level
- 8 neighbors in the current DoG level
- 9 samples in the next DoG level

Total comparison neighbors:

$$
26
$$

This lesson implements that 26-neighbor test explicitly.

## 10. Raw extrema are not final SIFT keypoints

The output of Part 1 is deliberately called **raw scale-space extrema**.

A full SIFT pipeline still needs to:

1. refine location in (x,y,\sigma)
2. reject low-contrast points
3. reject edge-like responses
4. assign one or more orientations
5. build descriptors

Those are covered in Parts 2 and 3.

## Implementation assumptions

For teaching clarity, this implementation:

- uses the original image resolution as octave 0
- treats the input image as the unblurred starting image
- applies (sigma_0) to create the first Gaussian level
- does not double the input image
- uses `CV_32F` for Gaussian and DoG data

The original SIFT paper and production implementations may include assumptions about input-image blur and image doubling. Those details will be discussed as implementation variants rather than hidden inside the code.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_sift_scale_space
```

## Run

```bash
./build/cv9x_sift_scale_space \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/sift_scale_space \
  --octaves 4 \
  --intervals 3 \
  --sigma 1.6 \
  --contrast-threshold 5
```

## Outputs

The lesson creates:

```text
build/sift_scale_space/
├── gaussian/
│   ├── gaussian_o00_l00.png
│   ├── gaussian_o00_l01.png
│   └── ...
├── dog/
│   ├── dog_o00_l00.png
│   ├── dog_o00_l01.png
│   └── ...
└── scale_space_extrema.png
```

The numbered octave/layer names make the scale progression easy to inspect.

## Deterministic self-test

```bash
./build/cv9x_sift_scale_space --self-test
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- (k^s = 2)
- Gaussian layer count is (s+3)
- DoG layer count is (s+2)
- incremental sigma reproduces the target absolute sigma
- a constant image produces near-zero DoG response
- a constant image produces no scale-space extrema
- octave dimensions halve correctly
- an impulse peak decreases as Gaussian scale increases
- DoG equals subtraction of adjacent Gaussian images
- the 26-neighbor test detects explicit positive and negative extrema

## Common failure cases

### Blurring every level with the absolute sigma

This over-blurs the pyramid because each image is already blurred.

Use the incremental sigma:

$$
\sqrt{\sigma_{target}^2-\sigma_{prev}^2}
$$

### Using `pyrDown()` blindly between SIFT octaves

`pyrDown()` adds its own low-pass filtering. That can break the sigma bookkeeping if you also assume the chosen Gaussian level already has the correct octave-transition blur.

### Storing DoG in unsigned 8-bit

Negative responses are meaningful. Keep the algorithm data signed or floating point.

### Searching only within one DoG image

SIFT extrema are 3-D extrema across (x), (y), **and scale**.

### Calling raw extrema "SIFT keypoints"

They have not yet passed subpixel localization, contrast rejection, edge rejection, or orientation assignment.

## Engineering notes

### Scale metadata is part of the interface

A production representation should track:

- octave index
- scale layer
- image resolution
- absolute sigma
- response
- coordinate mapping to the original image

### Avoid hidden blur

If multiple functions each apply smoothing, the true effective sigma can become unclear. Treat blur bookkeeping as explicit algorithm state.

### Memory grows quickly

Each octave stores several floating-point images. Later production lessons can discuss reusing buffers and streaming adjacent scale levels when the complete pyramid is not required simultaneously.

## Exercises

1. Add input-blur compensation using an assumed camera/input sigma.
2. Add optional 2x input upsampling and compare extrema.
3. Print the absolute sigma for every Gaussian level.
4. Generate a synthetic Gaussian blob and plot its center DoG response across scale.
5. Compare DoG images with a scale-normalized Laplacian-of-Gaussian response.
6. Benchmark memory usage as octave count and image resolution grow.

## Next lesson

Episode 11 covers **SIFT keypoint localization and orientation**:

- subpixel / subscale refinement
- low-contrast rejection
- Hessian edge-response rejection
- orientation histogram
- dominant orientation assignment

See [VIDEO.md](VIDEO.md).
