# Linear Filtering: Box Filter vs Gaussian Filter

This is the canonical linear-filtering lesson for Module 03 / Episode 04 of **9x Computer Vision**.

The previous lesson introduced convolution and correlation from first principles. This lesson uses that foundation to answer a practical question:

> If both filters smooth an image, why does Gaussian filtering usually look better than a box filter, and why does separability matter in production C++?

## Learning objectives

By the end of this lesson you should be able to:

- explain what makes an image filter linear
- construct a normalized box kernel
- construct a Gaussian kernel from `sigma`
- explain why a Gaussian gives more weight to nearby pixels
- compare box and Gaussian smoothing using matched parameters
- explain Gaussian separability and its computational advantage
- validate custom kernels against OpenCV reference implementations
- reason about border handling and its effect on output pixels
- measure filter latency without confusing one desktop measurement with a deployment guarantee

## Prerequisites

Complete these first:

1. [Point operations](../../01_PointFiltering/Readme.md)
2. [Convolution from first principles](../../00_ConvolutionFirstPrinciples/README.md)

## 1. What makes a filter linear?

A filter `F` is linear when it satisfies superposition:

\[
F(aI_1 + bI_2) = aF(I_1) + bF(I_2)
\]

for images `I1`, `I2` and scalar constants `a`, `b`.

Spatial filtering with a fixed kernel is linear because every output pixel is a weighted sum of input pixels.

For a kernel `K` centered at `(x, y)`:

\[
O(x,y) = \sum_i \sum_j K(i,j) I(x+i,y+j)
\]

The lesson self-test checks this property numerically.

## 2. The box filter

A normalized `k x k` box kernel gives every pixel in the neighborhood the same weight:

\[
K_{box}(i,j) = \frac{1}{k^2}
\]

For a 3x3 filter:

\[
K_{box} = \frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
\]

The weights sum to one, so a constant-intensity image remains constant away from border effects.

### Intuition

A box filter says:

> Every sample inside the window is equally trustworthy.

That is simple and useful, but the abrupt edge of the window can create a less natural smoothing response.

## 3. The Gaussian filter

A 2-D Gaussian is:

\[
G(x,y) = \frac{1}{2\pi\sigma^2}
\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
\]

After discretization and normalization, nearby samples receive more weight than samples near the edge of the kernel.

`\sigma` controls the spread:

- small sigma: weight is concentrated near the center
- larger sigma: smoothing extends over a wider neighborhood

A common conceptual rule is that most Gaussian energy lies within roughly `3 sigma` of the center, so a kernel radius near `3 sigma` is often a useful starting point. In this lesson, kernel size and sigma are explicit CLI parameters so you can experiment with them independently.

## 4. Why Gaussian smoothing often looks better

A box filter has uniform weights followed by a sharp cutoff at the kernel boundary.

A Gaussian has smoothly decaying weights. This generally reduces the abrupt spatial weighting introduced by the box filter and tends to produce a more natural low-pass smoothing response.

Neither filter is universally "better." A box filter can be completely appropriate when:

- computational simplicity matters
- equal weighting is intended
- a fast local average is the actual requirement

Gaussian smoothing is commonly chosen when the goal is controlled suppression of high-frequency noise before later operations such as gradients, scale-space analysis, or feature detection.

## 5. Kernel normalization

For a smoothing kernel, the sum should normally be one:

\[
\sum_i \sum_j K(i,j) = 1
\]

This lesson prints the sum of both kernels and tests it.

If a smoothing kernel is not normalized, it can unintentionally brighten or darken the image.

## 6. Gaussian separability

The 2-D Gaussian is separable:

\[
G(x,y) = g(x)g(y)
\]

That means a 2-D Gaussian operation can be implemented as:

1. one 1-D horizontal filter
2. one 1-D vertical filter

instead of applying the full 2-D kernel directly.

For a `k x k` kernel, a direct implementation performs roughly `k^2` multiply-accumulate operations per output pixel. A separable implementation needs roughly `2k`.

This is one of the most important engineering ideas in classical vision: **use mathematical structure to reduce computation without changing the intended result.**

The box kernel is also separable. Optimized box-filter implementations may additionally use running sums or related techniques, so do not assume a simple `k^2` implementation describes the performance of a production library.

## 7. Border handling

The kernel extends outside the image near an edge, so the implementation needs a border policy.

This lesson supports:

- `reflect101` — mirror pixels without repeating the edge sample
- `replicate` — repeat the nearest edge sample

Border handling is part of the algorithm contract. Two correct implementations can disagree near the image edge if they use different border rules.

## 8. What the C++ reference program compares

The program builds and compares:

### Box filter

- explicit normalized 2-D box kernel + `cv::filter2D`
- `cv::boxFilter` reference

### Gaussian filter

- explicit 2-D Gaussian kernel + `cv::filter2D`
- separable 1-D Gaussian + `cv::sepFilter2D`
- `cv::GaussianBlur` reference

The program keeps computation in `CV_32F` while validating numerical equivalence.

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_linear_filtering
```

Run the deterministic tests:

```bash
./build/cv9x_linear_filtering --self-test
```

or through CTest:

```bash
ctest --test-dir build --output-on-failure
```

## Run the lesson

```bash
./build/cv9x_linear_filtering \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/linear_filter_results \
  --kernel-size 5 \
  --sigma 1.2 \
  --border reflect101 \
  --iterations 50
```

Try a larger kernel:

```bash
./build/cv9x_linear_filtering \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/linear_filter_results_11x11 \
  --kernel-size 11 \
  --sigma 2.0 \
  --border replicate \
  --iterations 100
```

## Outputs

The executable writes:

- `box.png`
- `gaussian_2d.png`
- `gaussian_separable.png`
- `gaussian_opencv.png`
- `box_vs_gaussian_difference.png`
- `gaussian_2d_vs_separable_difference.png`

It also prints:

- box-kernel sum
- Gaussian-kernel sum
- maximum numerical difference between equivalent implementations
- the kernels for sizes up to 9x9
- average runtime for several OpenCV implementations

## What to look for

### Box vs Gaussian

Compare `box.png` and `gaussian_opencv.png` around:

- sharp boundaries
- small bright/dark regions
- fine texture

Increase the kernel size and observe how quickly detail is removed.

### 2-D vs separable Gaussian

`gaussian_2d.png` and `gaussian_separable.png` should be numerically extremely close. The difference visualization should contain little or no visible structure apart from floating-point effects.

## Self-test coverage

`--self-test` uses synthetic matrices and verifies:

- normalized box-kernel sum
- normalized Gaussian-kernel sum
- `filter2D` box result matches `boxFilter`
- 2-D Gaussian matches separable Gaussian
- separable Gaussian matches `GaussianBlur`
- the linearity property holds
- both `reflect101` and `replicate` border modes are covered

No external image file is required for CI.

## Engineering notes

### Keep a higher-precision intermediate type

The reference path converts grayscale input to `CV_32F` before filtering. This avoids hiding small implementation differences behind 8-bit rounding while we validate the algorithms.

### Kernel size and sigma are different controls

Kernel size decides how many samples are available. Sigma decides how quickly Gaussian weights decay. Treating them as interchangeable can produce surprising results.

### Separability reduces work, not mathematical intent

The separable form is not an approximation of the 2-D Gaussian used here. It is another way to compute the same separable kernel, subject to floating-point rounding.

### Measure the target system

The built-in timing loop is useful for exploration, but it is not a real-time guarantee and should not be used as deployment evidence. CPU architecture, OpenCV build options, SIMD paths, cache behavior, image size, thread configuration, frequency scaling, and system load can all change timing.

### Benchmark equivalent work

A performance comparison is meaningful only when image type, image size, kernel parameters, border handling, and output requirements are comparable.

## Common mistakes

1. Using an even kernel size without defining an anchor convention.
2. Forgetting to normalize a smoothing kernel.
3. Comparing filters that use different border policies.
4. Comparing Gaussian kernels with different sigma values and calling it a performance comparison.
5. Assuming a larger kernel always means proportionally larger runtime in an optimized library.
6. Converting to 8-bit before numerical validation and hiding small errors through rounding.
7. Saying `GaussianBlur` is faster because of one desktop timing sample.

## Exercises

### Exercise 1 — Sigma sweep

Keep the kernel size at 9 and run sigma values `0.7`, `1.5`, `2.5`, and `4.0`. Explain what changes and why.

### Exercise 2 — Kernel-size sweep

Hold sigma constant and test 3x3, 5x5, 9x9, and 15x15 kernels. At what point does increasing the window add little visible change for your chosen sigma?

### Exercise 3 — Border behavior

Create an image with a bright object touching the image boundary. Compare `reflect101` and `replicate`.

### Exercise 4 — Impulse response

Generate a black image with one white center pixel. Run both filters. The output directly visualizes each filter's kernel shape.

### Exercise 5 — Noise experiment

Add synthetic Gaussian noise to an image. Compare the amount of smoothing and edge loss produced by the box and Gaussian filters.

### Exercise 6 — Performance reasoning

Compare the direct 2-D Gaussian and separable Gaussian timing for multiple kernel sizes. Explain the trend using operation counts, but also identify why the measured ratio may not exactly equal the theoretical ratio.

## Next lesson

The next module moves into **image pyramids and multi-scale processing**, where Gaussian smoothing becomes part of a larger scale-space pipeline.

## Video

See [VIDEO.md](VIDEO.md) for the Episode 04 recording plan.
