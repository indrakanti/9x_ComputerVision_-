# Module 06 — Image Derivatives & Edge Detection

This lesson connects the gradient lesson to full edge-detection pipelines.

The central idea is that an **edge is a rapid spatial change**, but there are multiple mathematical ways to expose that change:

- first derivative: look for a large gradient magnitude
- second derivative: look for a sign change / zero crossing

This lesson intentionally stops before Canny. Episode 08 will build Canny from its individual stages.

## Learning objectives

You should be able to:

- distinguish first and second image derivatives
- explain why a step edge produces a first-derivative peak
- explain why a second derivative changes sign around an edge
- preserve signed Laplacian values rather than immediately taking absolute value
- build an edge map from gradient magnitude and a threshold
- detect approximate Laplacian zero crossings
- explain why derivatives amplify noise
- explain why Gaussian smoothing is commonly applied before differentiation
- validate OpenCV's Laplacian against an explicit kernel

## Prerequisites

- convolution
- Gaussian filtering
- Sobel / Scharr gradients

## 1. What is an edge?

Consider a one-dimensional intensity profile containing a dark-to-bright step.

The image intensity (I(x)) changes rapidly around the transition.

The first derivative

[
\frac{dI}{dx}
]

produces a strong peak at the transition.

The second derivative

[
\frac{d^2I}{dx^2}
]

produces positive and negative lobes, with a sign transition near the edge.

That gives us two common edge ideas:

1. threshold the gradient magnitude
2. detect second-derivative zero crossings

## 2. First-derivative edge map

From the previous lesson:

[
M = \sqrt{G_x^2 + G_y^2}
]

A simple edge detector is:

[
E(x,y) =
\begin{cases}
255, & M(x,y) \ge T \\
0, & M(x,y) < T
\end{cases}
]

This is intentionally simple.

It does **not** yet perform:

- non-maximum suppression
- double thresholding
- hysteresis

Those are the reasons Canny produces thinner and better-connected edges.

## 3. Second derivative: Laplacian

The continuous Laplacian is

[
\nabla^2 I =
\frac{\partial^2 I}{\partial x^2}
+
\frac{\partial^2 I}{\partial y^2}
]

A common 4-neighbor discrete approximation is:

[
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
]

This lesson uses `cv::Laplacian(..., ksize=1)` and verifies that it matches this explicit kernel.

## 4. Why the Laplacian must stay signed

At an edge, the Laplacian typically has one sign on one side and the opposite sign on the other.

Calling `convertScaleAbs()` too early destroys the sign transition.

For algorithm work, this lesson stores the Laplacian as:

```text
CV_32F
```

Only the debug PNG is remapped around neutral gray.

## 5. Zero crossings

A simple zero-crossing detector examines neighboring Laplacian samples.

An edge candidate exists when:

- the values have opposite signs
- their difference is large enough to exceed a contrast threshold

The contrast threshold prevents tiny noisy sign changes from being treated as meaningful edges.

This is a teaching implementation, not a replacement for a complete Marr-Hildreth or Canny implementation.

## 6. Derivatives amplify noise

Differentiation emphasizes high-frequency content.

Noise is often high-frequency content.

That means derivatives can respond strongly to noise even when the underlying image has no real object boundary.

The pipeline therefore supports Gaussian smoothing:

[
I_s = G_\sigma * I
]

followed by differentiation.

The self-test constructs deterministic noise and verifies that Gaussian smoothing reduces the mean absolute Laplacian response.

## 7. Why normalization is not an edge threshold

A common demo pattern is:

1. compute gradient
2. normalize it to 0–255
3. display it

That is useful for visualization, but it makes the display scale depend on the current image.

A production threshold should operate on a defined computational quantity, not on a per-image visualization normalization.

This lesson therefore:

- thresholds the original floating-point magnitude
- normalizes only the saved visualization

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_edge_derivatives
```

## Run

```bash
./build/cv9x_edge_derivatives \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/edge_derivative_results \
  --sigma 1.0 \
  --gradient-threshold 100 \
  --zero-crossing-threshold 15 \
  --border reflect101
```

## Outputs

```text
smoothed.png
gradient_magnitude.png
gradient_edges.png
laplacian_signed.png
laplacian_absolute.png
laplacian_zero_crossings.png
```

Important:

- `gradient_magnitude.png` is normalized for viewing
- `gradient_edges.png` comes from the real floating-point magnitude threshold
- `laplacian_signed.png` is a visualization centered at gray
- internal Laplacian values remain signed

## Deterministic validation

```bash
./build/cv9x_edge_derivatives --self-test
ctest --test-dir build --output-on-failure
```

The test checks:

- constant image -> zero first derivative
- constant image -> zero Laplacian
- constant image -> no zero crossings
- linear ramp -> nonzero first derivative
- linear ramp -> zero interior Laplacian
- step edge -> thresholded gradient edge
- step edge -> positive and negative Laplacian lobes
- step edge -> zero crossing
- higher threshold cannot increase edge count
- explicit 4-neighbor Laplacian == OpenCV `ksize=1`
- Gaussian smoothing reduces derivative response to deterministic noise

## First derivative vs second derivative

| Property | First derivative | Second derivative |
|---|---|---|
| Example | Sobel magnitude | Laplacian |
| Edge cue | local maximum / high magnitude | zero crossing |
| Sign useful? | yes for direction/polarity | essential for zero crossings |
| Noise sensitivity | high | typically even higher |
| Needs smoothing? | often | very often |
| Produces thin edge automatically? | no | not necessarily |

## Common failure cases

### Converting Laplacian to absolute value before zero-crossing detection

You lose the sign information needed for the zero crossing.

### Thresholding a normalized visualization

The numeric meaning of the threshold changes whenever the image's min/max changes.

### Skipping smoothing on noisy data

Derivative operators can turn sensor noise into false edges.

### Assuming every zero crossing is a real edge

Tiny oscillations can generate sign changes. Use a response/contrast criterion.

### Calling this Canny

It is not. Canny adds non-maximum suppression, double thresholding, and connectivity-based hysteresis.

## Engineering notes

A useful pipeline contract is:

```text
input
 -> optional smoothing
 -> signed derivative fields
 -> derived edge evidence
 -> threshold / selection logic
 -> binary edge representation
```

Do not collapse those stages into one display-oriented image if later components need the original evidence.

For embedded or real-time pipelines, also define:

- input bit depth
- derivative output type
- kernel size
- smoothing sigma
- border mode
- threshold units
- worst-case latency
- memory ownership

## Exercises

1. Add an 8-neighbor Laplacian kernel and compare it to the 4-neighbor version.
2. Sweep Gaussian sigma and record the edge count for a noisy image.
3. Sweep gradient threshold and plot edge-pixel count vs threshold.
4. Implement zero-crossing detection using opposite pixels across x/y rather than any 8-neighbor.
5. Add a synthetic blurred step and study how edge localization changes.

## Next lesson

Episode 08 will implement **Canny from scratch**:

1. Gaussian smoothing
2. Sobel gradients
3. gradient magnitude + direction
4. non-maximum suppression
5. double threshold
6. hysteresis edge tracking

See [VIDEO.md](VIDEO.md).
