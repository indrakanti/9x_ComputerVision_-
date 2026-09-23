# Module 08 — SIFT Part 3: 128-D Descriptor & Feature Matching

Part 1 found scale-space extrema.

Part 2 localized them, rejected unstable responses, and assigned orientation.

Part 3 answers the final question:

> How do we turn an oriented keypoint into a numeric vector that can be compared with a keypoint in another image?

This lesson implements the classic **4×4 spatial cells × 8 orientation bins = 128 dimensions** descriptor, then implements Euclidean matching and Lowe's ratio test.

For the real-image demo, OpenCV SIFT is used only to provide oriented keypoints. The descriptor and matcher are implemented explicitly so this lesson can focus on the representation and comparison stages.

## Learning objectives

By the end of this lesson you should be able to:

- explain why an oriented keypoint still needs a descriptor
- explain the 4×4×8 SIFT descriptor layout
- rotate the sampling frame by the keypoint orientation
- compute gradient magnitude and relative orientation
- apply Gaussian spatial weighting
- distribute votes by trilinear interpolation
- normalize, clip, and renormalize the descriptor
- compute Euclidean descriptor distance
- explain nearest-neighbor ambiguity
- implement Lowe's ratio test
- distinguish descriptor matching from geometric verification

## 1. Descriptor layout

The classic SIFT descriptor divides the keypoint neighborhood into:

$$
4 \times 4
$$

spatial cells.

Each cell stores an orientation histogram with:

$$
8
$$

bins.

Therefore:

$$
4 \times 4 \times 8 = 128
$$

descriptor values.

## 2. Orientation-aligned frame

The descriptor is built relative to the keypoint orientation assigned in Part 2.

For an image-space offset $(dx,dy)$ and keypoint angle $\theta_k$, the lesson rotates the sample into the keypoint frame:

$$
x' = \cos\theta_k\,dx + \sin\theta_k\,dy
$$

$$
y' = -\sin\theta_k\,dx + \cos\theta_k\,dy
$$

This is the core reason the descriptor can compare similar local structure after image rotation.

## 3. Gradient evidence

At each sample:

$$
G_x = I(x+1,y)-I(x-1,y)
$$

$$
G_y = I(x,y+1)-I(x,y-1)
$$

Magnitude:

$$
m = \sqrt{G_x^2+G_y^2}
$$

Gradient angle:

$$
\theta_g = \operatorname{atan2}(G_y,G_x)
$$

The descriptor uses relative orientation:

$$
\theta_{rel} = \theta_g - \theta_k
$$

wrapped to $[0,360)$.

## 4. Scale-dependent descriptor window

The keypoint size determines the spatial support.

This lesson uses a descriptor histogram width proportional to keypoint sigma:

```text
hist_width = 3 × sigma
```

The four spatial cells then cover a scale-dependent neighborhood.

This is what connects the descriptor to the scale selected by the detector.

## 5. Gaussian weighting

Samples near the descriptor center should contribute more than distant samples.

The implementation uses a Gaussian spatial weight over the descriptor window.

This reduces the influence of gradients near the support boundary.

## 6. Trilinear interpolation

A gradient sample should not jump discontinuously between histogram bins as it moves slightly.

The implementation distributes each gradient vote across:

- two neighboring x cells
- two neighboring y cells
- two neighboring orientation bins

That is trilinear interpolation across the descriptor's $(x,y,\theta)$ histogram grid.

Each sample may therefore contribute to up to:

$$
2 \times 2 \times 2 = 8
$$

descriptor bins.

## 7. Normalize → clip → renormalize

Raw gradient magnitudes are sensitive to illumination contrast.

SIFT reduces that sensitivity by first L2-normalizing:

$$
\mathbf{d}
\leftarrow
\frac{\mathbf{d}}{\|\mathbf{d}\|_2}
$$

Then large components are clipped:

$$
d_i \leftarrow \min(d_i, 0.2)
$$

Finally the vector is normalized again.

This limits domination by a few very strong gradients.

The CLI exposes the clipping value:

```text
--clip 0.2
```

## 8. Descriptor distance

Two descriptors are compared using Euclidean distance:

$$
D(\mathbf{a},\mathbf{b})
=
\sqrt{
\sum_i (a_i-b_i)^2
}
$$

Small distance means similar local gradient structure.

But the nearest descriptor is not automatically a trustworthy match.

## 9. Why the second-nearest neighbor matters

Suppose one query descriptor has:

```text
nearest distance        = 0.20
second-nearest distance = 0.22
```

The match is ambiguous.

Another query has:

```text
nearest distance        = 0.20
second-nearest distance = 0.80
```

That nearest neighbor is much more distinctive.

The ratio is:

$$
r =
\frac{d_1}{d_2}
$$

The lesson accepts a match when:

$$
d_1 < t d_2
$$

with default:

```text
t = 0.75
```

## 10. What the ratio test does not prove

The ratio test checks **descriptor distinctiveness**.

It does not prove that a set of matches is geometrically consistent.

A complete image-matching pipeline still needs:

- homography or essential/fundamental-matrix estimation
- RANSAC or another robust estimator
- inlier/outlier classification

That becomes the feature-matching / geometry lesson.

## Real-image demo contract

To isolate Part 3, the demo obtains oriented keypoints using:

```cpp
cv::SIFT::detect(...)
```

It does **not** use:

```cpp
cv::SIFT::compute(...)
cv::SIFT::detectAndCompute(...)
cv::BFMatcher
```

The following are manual in this lesson:

- descriptor construction
- descriptor normalization/clipping
- L2 distance
- nearest / second-nearest search
- ratio test
- match visualization

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_sift_descriptor_matching
```

## Run

Use any two related grayscale images:

```bash
./build/cv9x_sift_descriptor_matching \
  --image1 path/to/image1.png \
  --image2 path/to/image2.png \
  --output-dir build/sift_part3 \
  --max-features 400 \
  --ratio 0.75 \
  --clip 0.2
```

Output:

```text
sift_manual_descriptor_matches.png
```

## Deterministic self-test

```bash
./build/cv9x_sift_descriptor_matching --self-test
ctest --test-dir build --output-on-failure
```

The self-test verifies:

- descriptor length is exactly 128
- a nonzero descriptor is L2-normalized
- multiplicative contrast scaling produces nearly the same normalized descriptor
- orientation alignment makes equivalent horizontal/vertical gradient structure similar
- clipping is followed by renormalization
- identical descriptors have zero L2 distance
- the ratio test accepts a distinctive nearest neighbor
- the ratio test rejects an ambiguous nearest neighbor
- a flat patch produces a zero descriptor

## Common failure cases

### Building the descriptor in image coordinates

Then rotating the camera/image changes the descriptor substantially.

Use the keypoint-oriented coordinate frame.

### Using absolute gradient direction

The descriptor bins orientation relative to the keypoint's dominant orientation.

### Hard-assigning every sample to one cell and one orientation bin

Small image motions cause unstable jumps.

Interpolate across neighboring spatial/orientation bins.

### Normalizing only once

The standard robustness step is normalize, clip large components, then normalize again.

### Accepting the closest descriptor unconditionally

Nearest-neighbor matching alone is often ambiguous.

Use the nearest-vs-second-nearest relationship.

### Calling ratio-test matches geometric inliers

Descriptor similarity and geometric consistency are different problems.

## Engineering notes

For production pipelines, a descriptor interface should define:

- descriptor dimension
- scalar type
- normalization convention
- clipping convention
- distance metric
- keypoint coordinate/scale/orientation convention
- invalid/zero descriptor behavior
- maximum feature count
- deterministic matching/tie policy

For embedded systems, 128 floating-point values per feature also have a real memory and bandwidth cost.

For example:

```text
128 floats × 4 bytes = 512 bytes / descriptor
1000 descriptors      = ~500 KiB
```

before metadata.

That motivates compact binary descriptors such as BRIEF/ORB later in the course.

## Exercises

1. Compare manual descriptors against `cv::SIFT::compute()` for the same keypoints.
2. Sweep the ratio threshold from 0.5 to 0.95 and plot accepted-match count.
3. Add mutual nearest-neighbor / cross-check matching.
4. Quantize the normalized descriptor to 8-bit values and compare matching behavior.
5. Benchmark brute-force manual matching as feature count grows.
6. Add RANSAC homography filtering to separate descriptor matches from geometric inliers.

## Next lesson

Episode 13 compares **ORB vs SIFT** and introduces binary descriptors, Hamming distance, speed, memory, and embedded-system tradeoffs.

The following matching lesson then adds **ratio test + RANSAC geometric verification** as a complete correspondence pipeline.

See [VIDEO.md](VIDEO.md).
