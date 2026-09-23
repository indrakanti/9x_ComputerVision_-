# Module 08 — SIFT Part 2: Keypoint Localization, Edge Rejection & Orientation

Part 1 produced **raw Difference-of-Gaussian extrema**.

Those integer pixel / integer scale candidates are not yet reliable SIFT keypoints.

Part 2 turns them into stable, oriented features by applying four steps:

1. Taylor-based subpixel and subscale localization
2. low-contrast rejection
3. edge-response rejection using the spatial Hessian
4. dominant-orientation assignment

The descriptor itself is intentionally deferred to Part 3.

## Learning objectives

By the end of this lesson you should be able to:

- explain why integer DoG extrema need refinement
- derive the 3-D Taylor approximation used for SIFT localization
- compute the offset in $x$, $y$, and scale
- explain interpolated DoG contrast
- explain why strong edge responses make poor keypoints
- derive the Hessian principal-curvature ratio test
- build a weighted orientation histogram
- understand why one localized keypoint may receive multiple orientations
- distinguish localization scale from original-image coordinates

## Prerequisite

Complete [SIFT Part 1 — Scale Space & Difference of Gaussians](../00_ScaleSpaceDoG/README.md).

## 1. Why refine a DoG extremum?

Part 1 found extrema only on the discrete sampling grid:

```text
integer x
integer y
integer DoG layer
```

The real extremum may lie between samples.

SIFT approximates the local DoG surface with a Taylor expansion and estimates the fractional offset.

## 2. Three-dimensional Taylor model

Let the candidate location be

$$
\mathbf{x} =
\begin{bmatrix}
x \\
y \\
s
\end{bmatrix}
$$

where $s$ is scale-layer position.

Around the integer candidate, approximate the DoG function as

$$
D(\mathbf{x}+\Delta)
\approx
D +
\nabla D^T \Delta +
\frac{1}{2}\Delta^T H\Delta
$$

where:

- $\nabla D$ is the 3-D gradient in $x,y,s$
- $H$ is the 3x3 Hessian
- $\Delta$ is the fractional offset

At the local extremum, the derivative is zero:

$$
\nabla D + H\Delta = 0
$$

Therefore:

$$
\Delta = -H^{-1}\nabla D
$$

The implementation computes these derivatives with centered finite differences.

## 3. Iterative relocation

If any component of $\Delta$ is larger than roughly half a sample, the integer candidate is moved to the neighboring sample and the Taylor model is recomputed.

This lesson allows up to five relocation iterations.

A candidate is rejected when it:

- leaves the valid 3-D neighborhood
- encounters a singular Hessian
- does not converge

## 4. Interpolated contrast

Once the offset converges, the DoG value at the refined position is approximated by

$$
D(\hat{\mathbf{x}})
\approx
D +
\frac{1}{2}
\nabla D^T\Delta
$$

Low-magnitude responses are unstable and noise-sensitive, so the implementation rejects them with:

```text
--contrast-threshold
```

The threshold in this teaching implementation is expressed directly in the raw `CV_32F` DoG units.

That is intentionally different from copying a numeric threshold from another SIFT implementation whose image normalization and internal scaling may differ.

## 5. Why edge-like extrema are rejected

A DoG response can be very strong along a ridge or edge.

But localization along that edge is ambiguous.

SIFT therefore examines the **2-D spatial Hessian** of the DoG layer:

$$
H_{xy} =
\begin{bmatrix}
D_{xx} & D_{xy} \\
D_{xy} & D_{yy}
\end{bmatrix}
$$

Let its principal curvatures be $\alpha$ and $\beta$.

Instead of explicitly computing the two eigenvalues, SIFT uses:

$$
\frac{\operatorname{Tr}(H_{xy})^2}
     {\det(H_{xy})}
$$

If the allowed curvature ratio is $r$, the candidate must satisfy:

$$
\frac{\operatorname{Tr}(H_{xy})^2}
     {\det(H_{xy})}
<
\frac{(r+1)^2}{r}
$$

and the determinant must be positive.

The default lesson value is:

```text
r = 10
```

## 6. Refined scale

If the refined DoG layer is $l + \Delta_s$, then the scale within the octave is

$$
\sigma =
\sigma_0
k^{l+\Delta_s}
$$

where

$$
k=2^{1/s}
$$

The original-image scale is this octave-relative sigma multiplied by $2^{octave}$.

## 7. Orientation assignment

A scale-invariant keypoint is not yet rotation invariant.

SIFT assigns orientation from local image gradients measured in the Gaussian image corresponding to the keypoint scale.

For each sample in a neighborhood:

$$
m(x,y) =
\sqrt{
(L(x+1,y)-L(x-1,y))^2 +
(L(x,y+1)-L(x,y-1))^2
}
$$

and

$$
\theta(x,y) =
\operatorname{atan2}(L_y,L_x)
$$

The implementation uses a 36-bin histogram over $0$ to $360$ degrees.

## 8. Gaussian weighting

Samples near the keypoint should contribute more than distant samples.

SIFT therefore applies a Gaussian spatial weight.

This lesson uses orientation-window sigma:

$$
\sigma_{ori} = 1.5\sigma
$$

and samples approximately within:

$$
3\sigma_{ori}
$$

## 9. Histogram smoothing and peak interpolation

The histogram is circular.

The implementation:

1. distributes gradient votes between adjacent orientation bins
2. smooths the circular histogram repeatedly
3. finds local maxima
4. fits a parabola around each peak for fractional-bin orientation

The strongest peak defines the dominant orientation.

Other local peaks above a fraction of the strongest peak can create additional oriented keypoint instances.

Default:

```text
--orientation-peak-ratio 0.8
```

That behavior is important: **one localized point can legitimately produce multiple SIFT orientations**.

## 10. Coordinate conventions

The implementation stores keypoint coordinates in the current octave.

For visualization:

$$
x_{base} = x_{octave} 2^{octave}
$$

$$
y_{base} = y_{octave} 2^{octave}
$$

Orientation uses image coordinates, so positive $y$ points downward.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_sift_localization_orientation
```

## Run

```bash
./build/cv9x_sift_localization_orientation \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/sift_part2 \
  --octaves 4 \
  --intervals 3 \
  --sigma 1.6 \
  --candidate-threshold 1 \
  --contrast-threshold 2 \
  --edge-ratio 10 \
  --orientation-bins 36 \
  --orientation-peak-ratio 0.8
```

## Outputs

```text
01_raw_scale_space_extrema.png
02_localized_keypoints.png
03_oriented_keypoints.png
```

The executable also reports how many candidates were rejected because of:

- invalid neighborhood
- singular Hessian
- failed convergence
- low contrast
- edge-like curvature

## Deterministic self-test

```bash
./build/cv9x_sift_localization_orientation --self-test
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- a synthetic 3-D quadratic extremum is localized to known fractional $x$, $y$, and scale coordinates
- the same localization path rejects a low-contrast extremum
- an isotropic blob passes the curvature-ratio check
- an anisotropic ridge fails the curvature-ratio check
- a horizontal intensity ramp produces orientation near 0 degrees
- a vertical intensity ramp produces orientation near 90 degrees
- a flat patch produces no orientation
- three scale intervals double sigma

## What this lesson deliberately does not do

Part 2 stops after orientation assignment.

It does not yet construct SIFT's 128-element descriptor.

That requires:

- rotating the descriptor frame to the assigned orientation
- dividing the neighborhood into spatial cells
- building local gradient histograms
- normalization
- clipping large values
- renormalization

Those are Part 3.

## Common failure cases

### Using the integer DoG sample as the final keypoint

You lose subpixel/subscale localization and may keep unstable candidates.

### Rejecting edges using gradient magnitude

SIFT's edge rejection is based on the **spatial second-derivative curvature ratio** of the DoG response.

### Computing orientation from the DoG image

Orientation is assigned from the Gaussian image at the keypoint scale.

### Forgetting that the orientation histogram is circular

Bin 0 and the final bin are neighbors.

### Keeping only one orientation unconditionally

Strong secondary peaks can represent legitimate multiple orientations.

### Copying OpenCV threshold numbers directly

Threshold meaning depends on image scaling and implementation normalization.

## Engineering notes

A production keypoint representation should define:

- octave
- discrete layer
- sublayer offset
- octave coordinates
- base-image coordinates
- octave sigma
- original-image sigma
- DoG contrast
- orientation
- rejection reason / diagnostics when debugging

This makes scale-space behavior observable rather than hidden inside one detector call.

## Exercises

1. Export accepted keypoints to CSV with octave, layer, sigma, contrast, and orientation.
2. Plot rejection counts as contrast threshold changes.
3. Sweep the curvature ratio $r$ and inspect ridge rejection.
4. Create a patch with two dominant gradient directions and verify multiple orientations.
5. Compare the oriented keypoints against `cv::SIFT::detect()`.
6. Replace nearest Gaussian-layer selection with interpolation between neighboring scale levels.

## Next lesson

Episode 12 / SIFT Part 3 builds the **128-element descriptor and feature matching pipeline**.

See [VIDEO.md](VIDEO.md).
