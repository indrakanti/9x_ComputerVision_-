# Image Pyramids and Multi-Scale Processing

This lesson builds directly on Gaussian filtering. Instead of processing an image at only one resolution, we construct a sequence of progressively smaller images and use that representation to reason about scale, aliasing, residual detail, reconstruction, and memory cost.

## Learning objectives

By the end of this lesson you should be able to:

- explain why downsampling without low-pass filtering can create aliasing;
- construct a Gaussian pyramid using repeated Gaussian reduction;
- explain what information each Gaussian level retains;
- construct a Laplacian pyramid as signed residual detail between adjacent Gaussian levels;
- reconstruct the original image from a Laplacian pyramid;
- explain why Laplacian residuals should use a signed/floating-point representation;
- estimate the memory cost of a multi-level 2x image pyramid;
- connect image pyramids to SIFT, optical flow, blending, stereo, and multi-scale detection.

## Prerequisites

Complete these lessons first:

1. point operations;
2. convolution from first principles;
3. box vs Gaussian linear filtering.

The important prerequisite is understanding that smoothing removes high-frequency image content before sampling at a lower spatial resolution.

---

## 1. Why one image scale is not enough

An object can occupy very different numbers of pixels depending on distance, focal length, image resolution, and crop size. A feature detector that works well when an object is 200 pixels wide may behave differently when the same object is only 20 pixels wide.

A pyramid gives the algorithm multiple spatial scales:

```text
G0   full resolution
G1   about 1/2 width and 1/2 height
G2   about 1/4 width and 1/4 height
G3   about 1/8 width and 1/8 height
...
```

Each level contains fewer pixels and less fine detail.

---

## 2. Downsampling is not just resize

Suppose we keep every second pixel in each direction. The output has only one quarter as many samples, so it cannot represent all spatial frequencies present in the original image.

If high-frequency content remains before decimation, it can fold into lower frequencies and create false patterns. This is **aliasing**.

The safe conceptual sequence is:

```text
input -> low-pass filter -> decimate
```

not simply:

```text
input -> throw away samples
```

The executable writes two first-level examples:

- `naive_downsample_level_1.png` using nearest-neighbor reduction;
- `gaussian_downsample_level_1.png` using `cv::pyrDown()`.

Use images containing checkerboards, thin lines, text, or repetitive textures to make the difference obvious.

---

## 3. Gaussian pyramid

Let `G_0` be the original image. A Gaussian pyramid level can be written conceptually as

\[
G_{l+1} = D(K * G_l)
\]

where:

- `K` is a low-pass Gaussian-like kernel;
- `*` denotes filtering;
- `D` denotes downsampling by approximately two in each direction.

OpenCV's `pyrDown()` performs the smoothing and reduction together. Its classic 5x5 separable weighting is based on the binomial vector

\[
[1,\ 4,\ 6,\ 4,\ 1]
\]

normalized in two dimensions.

The important engineering point is not the API call itself. It is the contract:

> remove spatial frequencies the coarser level cannot represent, then reduce the sampling rate.

### Pyramid dimensions

For a 640x480 image, repeated 2x reduction is approximately:

```text
G0: 640 x 480
G1: 320 x 240
G2: 160 x 120
G3:  80 x  60
```

For odd dimensions, OpenCV handles the extra pixel according to its pyramid size rules. Production code should never assume every source dimension is divisible by `2^levels`.

---

## 4. Pyramid memory intuition

Each 2x reduction has roughly one quarter the pixels of the previous level.

For a deep pyramid, the total number of stored pixels approaches

\[
1 + \frac{1}{4} + \frac{1}{16} + \frac{1}{64} + \cdots = \frac{4}{3}
\]

of the base image pixel count.

So a Gaussian pyramid can provide many spatial scales for roughly 33% more pixels than the original image, ignoring metadata, alignment, allocator overhead, and different element types.

The example prints the actual pixel ratio for the requested number of levels.

---

## 5. Laplacian pyramid

A Gaussian pyramid stores progressively smoother and smaller images. A Laplacian pyramid stores what was lost between adjacent scales.

For level `l`:

\[
L_l = G_l - E(G_{l+1})
\]

where `E` expands the next coarser Gaussian level back to the current level's size.

The final, smallest Gaussian image is stored as the pyramid base.

Conceptually:

```text
G0 ---- subtract expanded G1 ---> L0
G1 ---- subtract expanded G2 ---> L1
G2 ---- subtract expanded G3 ---> L2
G3 ------------------------------> base
```

The residual contains edges, texture, and other spatial detail that the coarser level could not reproduce.

### Why the residual must be signed

`G_l - E(G_{l+1})` can be negative or positive.

If you store that subtraction directly in an unsigned 8-bit image, negative values can be clipped and information can be destroyed. The canonical implementation therefore converts the working image to `CV_32F` before pyramid construction.

For display only, the program maps zero residual to mid-gray. Darker and lighter values show negative and positive detail.

---

## 6. Reconstruction

A correctly constructed Laplacian pyramid is not merely a visualization. It is a representation from which the original can be reconstructed.

Start from the smallest level and repeatedly expand and add the residual:

\[
\hat{G}_l = E(\hat{G}_{l+1}) + L_l
\]

The implementation performs this process back to level 0 and prints:

- maximum absolute reconstruction error;
- reconstruction RMSE.

Because the residual is generated from the same float-domain expand operation used during reconstruction, the expected error should be extremely small.

This is one of the strongest tests in the lesson. A pyramid that looks reasonable but cannot reconstruct its own input indicates a mathematical, type, sizing, or border-handling error.

---

## 7. Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_image_pyramids
```

Run the deterministic self-test:

```bash
./build/cv9x_image_pyramids --self-test
```

Or run all registered lesson tests:

```bash
ctest --test-dir build --output-on-failure
```

---

## 8. Run the lesson

Example using an existing course image:

```bash
./build/cv9x_image_pyramids \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/pyramid_results \
  --levels 4
```

The executable accepts 2 to 8 levels, subject to the input being large enough to support the requested depth.

---

## 9. Expected outputs

For four levels, the output directory contains files such as:

```text
gaussian_level_0.png
gaussian_level_1.png
gaussian_level_2.png
gaussian_level_3.png
laplacian_level_0.png
laplacian_level_1.png
laplacian_level_2.png
laplacian_base.png
naive_downsample_level_1.png
gaussian_downsample_level_1.png
reconstructed.png
reconstruction_difference.png
```

The terminal also prints each Gaussian level's dimensions, pyramid pixel ratio, maximum reconstruction error, and RMSE.

---

## 10. What the self-test checks

The `--self-test` path uses only synthetic in-memory data. It verifies:

1. the requested Gaussian level count;
2. expected dimensions for a known even-sized image;
3. matching Laplacian level count;
4. that a real Laplacian residual contains both negative and positive detail;
5. reconstruction of the original image to a maximum absolute error of at most `1e-4`;
6. preservation of a constant image through Gaussian reduction.

No external image file is required for CI.

---

## 11. Engineering notes

### Keep computation separate from visualization

The pyramid math stays in `CV_32F`. Conversion to `CV_8U` happens only when writing viewable PNG files.

### Do not clip residuals before reconstruction

Clipping is irreversible. Keep signed residual data until reconstruction or downstream processing is complete.

### Border rules are part of the algorithm

Near image edges, filtering requires values outside the image domain. `pyrDown()` and `pyrUp()` use OpenCV's pyramid border behavior. When reproducing the algorithm elsewhere, border policy must be treated as part of the interface contract.

### Pyramid depth has a cost

More levels provide coarser scales but also add latency, memory traffic, and storage. Stop when the spatial resolution is no longer useful to the downstream algorithm.

### Reuse buffers in production

This educational implementation uses `std::vector<cv::Mat>` for clarity. A real-time pipeline may pre-allocate pyramid buffers to control allocation latency and memory fragmentation.

---

## 12. Where pyramids appear in computer vision

Image pyramids are foundational to:

- SIFT and other scale-space feature methods;
- coarse-to-fine optical flow;
- multi-band image blending;
- stereo and correspondence search;
- template matching at multiple scales;
- classical image registration;
- multi-scale object detection;
- feature pyramid networks and modern detector architectures.

Later course modules will reuse this multi-scale reasoning rather than treating pyramids as an isolated trick.

---

## 13. Common failure cases

### Aliasing after downsampling

Cause: reducing resolution without enough low-pass filtering.

### Laplacian image looks almost black

Cause: displaying signed float residuals directly as unsigned pixels. Use a visualization mapping; do not change the underlying residual data.

### Reconstruction error is large

Check:

- whether the expand operation matches the one used when creating residuals;
- whether the target size is exact at each level;
- whether residuals were clipped or quantized;
- whether Gaussian/Laplacian levels were accidentally reordered.

### Requested pyramid depth fails

The source image became too small. Use fewer levels or a larger input image.

---

## 14. Exercises

### Exercise 1 — Make aliasing visible

Create a synthetic checkerboard or one-pixel stripe pattern and compare nearest-neighbor reduction against `pyrDown()`.

### Exercise 2 — Break reconstruction intentionally

Convert each Laplacian residual to `CV_8U` before reconstruction. Measure the error and explain why it increased.

### Exercise 3 — Storage budget

Compute the actual total pixel ratio for 2, 3, 4, 5, and 6 levels. Compare the measurements with the theoretical `4/3` limit.

### Exercise 4 — Multi-scale search

Choose a simple template and search for it independently at multiple Gaussian pyramid levels. Discuss how pyramid scale changes the effective object size.

---

## Next lesson

With multi-scale image representation established, the next course step is image gradients: Sobel, Scharr, gradient magnitude, and gradient direction.