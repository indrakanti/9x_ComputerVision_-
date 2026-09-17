# Video: Box Filter vs Gaussian Filter — Math, C++ and Real Results

YouTube: TBD

Target length: **18–24 minutes**

Episode: **04**

## Learning objectives

By the end of the video, viewers should be able to:

- define a linear spatial filter using superposition
- explain the difference between equal-weight box smoothing and Gaussian weighting
- construct normalized box and Gaussian kernels
- explain the role of sigma
- explain Gaussian separability
- understand why border handling must be part of a reproducible filter definition
- validate equivalent implementations numerically
- interpret timing results without over-generalizing them

## Searchable title

**Box Filter vs Gaussian Filter in C++ — Math, Separability & Performance**

Alternate title:

**Gaussian Blur Explained from First Principles — Why It Beats a Box Filter**

## Thumbnail concept

Split image:

- left: `BOX` over a flat 5x5 equal-weight kernel
- right: `GAUSSIAN` over a bright-center weighted kernel
- bottom text: `WHY DIFFERENT?`

Keep the image/result comparison large; avoid filling the thumbnail with equations.

## Visual hook / opening demo

Open with the same sharp synthetic image filtered using the same 9x9 support:

1. original
2. 9x9 box result
3. 9x9 Gaussian result

Zoom into one high-contrast boundary.

Opening line idea:

> Both of these filters average nearby pixels. Both are linear. Both reduce detail. But they do not behave the same, and the reason is visible directly inside their kernels.

Then show the two kernel heatmaps or printed matrices.

## Recording structure

### 00:00 — Box and Gaussian can use the same window but produce different results

Show:

- input
- box output
- Gaussian output

State the question:

> What changes when the neighborhood is the same size but the weights are different?

### 01:10 — What "linear filtering" actually means

Write:

\[
F(aI_1+bI_2)=aF(I_1)+bF(I_2)
\]

Then connect to the convolution/correlation lesson:

\[
O(x,y)=\sum_i\sum_j K(i,j)I(x+i,y+j)
\]

Key point:

- the output is a weighted sum
- with a fixed kernel, superposition holds

Mention that the lesson self-test checks linearity numerically.

### 03:00 — The box kernel

Show a 3x3 box kernel:

\[
\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}
\]

Explain:

- all positions receive equal weight
- normalization keeps DC gain at one
- abrupt cutoff at the edge of the neighborhood

Show 3x3, 5x5, and 9x9 results quickly.

### 05:30 — The Gaussian kernel

Write:

\[
G(x,y)=\frac{1}{2\pi\sigma^2}
\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
\]

Explain:

- center samples matter most
- influence decreases smoothly with distance
- sigma controls spread
- kernel size controls the finite window we keep

Show a printed 5x5 Gaussian matrix next to the 5x5 box matrix.

### 08:00 — Why Gaussian smoothing often looks more natural

Use an edge/texture crop.

Point out:

- box averaging gives all in-window samples equal authority
- Gaussian weighting decreases smoothly away from the center
- both are low-pass filters, but their frequency/spatial responses differ

Avoid claiming Gaussian is always better.

State when box filtering can be the right choice.

### 10:10 — Separability: the important engineering idea

Start with:

\[
G(x,y)=g(x)g(y)
\]

Draw:

`2-D k x k` -> `1-D horizontal` -> `1-D vertical`

Compare direct operation counts:

- full 2-D: roughly `k^2` multiply-accumulates per pixel
- separable: roughly `2k`

Emphasize:

> This is not changing the Gaussian. We are exploiting its mathematical structure to compute the same separable kernel more efficiently.

Mention that the box kernel is also separable and optimized box filtering may use running-sum methods.

### 12:45 — C++ walkthrough

Open:

`02_ImageFiltering/02_LinearFiltering/00_BoxVsGaussian/linear_filtering.cpp`

Walk through:

1. `makeBoxKernel()`
2. `makeGaussian1D()`
3. outer product creating `makeGaussian2D()`
4. `filter2D`
5. `boxFilter`
6. `sepFilter2D`
7. `GaussianBlur`
8. numerical comparisons
9. benchmark loop

Highlight that computation is kept in `CV_32F` while comparing implementations.

### 16:00 — Run the matched comparison

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_linear_filtering
```

Run:

```bash
./build/cv9x_linear_filtering \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/linear_filter_results \
  --kernel-size 5 \
  --sigma 1.2 \
  --border reflect101 \
  --iterations 50
```

Show terminal output:

- kernel sums
- max differences
- printed kernels
- average runtime

Then show:

- `box.png`
- `gaussian_opencv.png`
- `box_vs_gaussian_difference.png`
- `gaussian_2d_vs_separable_difference.png`

### 18:30 — Border handling changes the answer

Repeat with:

```bash
--border replicate
```

Show a crop at the image edge.

Key statement:

> Border policy is not a cosmetic implementation detail. It is part of the algorithm definition.

### 19:45 — Performance: what we can and cannot conclude

Show the timing table printed by the application.

Explain:

- timing is useful for local exploration
- do not make universal claims from one machine
- OpenCV may use SIMD, threading, specialized implementations, and architecture-specific optimizations
- deployment measurements must be repeated on the target platform

Connect to later course modules on latency, jitter, and production pipelines.

### 21:30 — Self-test and engineering confidence

Run:

```bash
./build/cv9x_linear_filtering --self-test
ctest --test-dir build --output-on-failure
```

Explain that the tests validate:

- kernel normalization
- box equivalence
- full 2-D vs separable Gaussian
- separable Gaussian vs `GaussianBlur`
- linearity
- border modes

### 22:30 — Exercise and next lesson

Give the impulse-response exercise:

> Make a black image with one white pixel at the center. Filter it with box and Gaussian kernels. The output is effectively a picture of the kernel itself.

Preview Episode 05:

> Next we will use Gaussian smoothing to build image pyramids and understand why multi-scale computer vision works.

## Equations / diagrams to show

### Linearity

\[
F(aI_1+bI_2)=aF(I_1)+bF(I_2)
\]

### Weighted neighborhood

\[
O(x,y)=\sum_i\sum_jK(i,j)I(x+i,y+j)
\]

### Box kernel

\[
K_{box}(i,j)=\frac{1}{k^2}
\]

### Gaussian

\[
G(x,y)=\frac{1}{2\pi\sigma^2}
\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
\]

### Separability

\[
G(x,y)=g(x)g(y)
\]

### Operation-count visual

```text
Direct 2-D Gaussian
k x k -> ~k^2 operations / pixel

Separable Gaussian
1 x k then k x 1 -> ~2k operations / pixel
```

## Code files used

Primary:

- `02_ImageFiltering/02_LinearFiltering/00_BoxVsGaussian/linear_filtering.cpp`

Background / legacy examples:

- `02_ImageFiltering/02_LinearFiltering/02_BoxFilter/boxfilter.cpp`
- `02_ImageFiltering/02_LinearFiltering/03_GaussianFilter/gaussian2d.cpp`

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_linear_filtering
./build/cv9x_linear_filtering --self-test
```

```bash
./build/cv9x_linear_filtering \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/linear_filter_results \
  --kernel-size 5 \
  --sigma 1.2 \
  --border reflect101 \
  --iterations 50
```

## Expected output

The executable should report very small numerical differences between mathematically equivalent paths and write:

- `box.png`
- `gaussian_2d.png`
- `gaussian_separable.png`
- `gaussian_opencv.png`
- `box_vs_gaussian_difference.png`
- `gaussian_2d_vs_separable_difference.png`

Do not hard-code expected benchmark numbers in the video because timing is machine-dependent.

## Engineering takeaway

End the technical section with four rules:

1. **Normalize smoothing kernels intentionally.**
2. **Specify border behavior.**
3. **Exploit separability when the math allows it.**
4. **Benchmark on the target system, not by assumption.**

## Exercise

Recommended viewer exercise:

Create an impulse image, run a 9x9 box filter and a 9x9 Gaussian filter, and explain why the two output patterns look like the corresponding kernels.

Advanced exercise:

Sweep kernel size and sigma independently and record both image-quality observations and runtime measurements.

## YouTube description

```text
Why does Gaussian blur usually look different from a box filter even when both use the same neighborhood?

In Episode 04 of 9x Computer Vision, we build both filters from their kernels, compare them using C++ and OpenCV, validate equivalent implementations, and show why Gaussian separability matters for performance.

You will learn:
- what makes a spatial filter linear
- normalized box-filter math
- Gaussian kernel math and sigma
- 2-D vs separable Gaussian filtering
- border handling
- numerical validation against OpenCV
- how to interpret filtering benchmarks responsibly

Lesson notes + source code:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/02_ImageFiltering/02_LinearFiltering/00_BoxVsGaussian

Full course repository:
https://github.com/indrakanti/9x_ComputerVision_-

Chapters:
00:00 Box vs Gaussian result
01:10 What linear filtering means
03:00 Box kernel
05:30 Gaussian kernel and sigma
08:00 Why the results differ
10:10 Gaussian separability
12:45 C++ walkthrough
16:00 Run the comparison
18:30 Border handling
19:45 Performance interpretation
21:30 Self-test
22:30 Exercise and next lesson

#ComputerVision #OpenCV #CPP #Linux #ImageProcessing #Robotics #EmbeddedAI
```

## Short / clip idea

### Title

**Why Gaussian Blur Is Faster Than a Naive 2-D Gaussian**

### 30–45 second structure

1. Show a `15 x 15` Gaussian kernel.
2. Say: "Naively, that is 225 weights per output pixel."
3. Replace it with a `1 x 15` horizontal vector and a `15 x 1` vertical vector.
4. Say: "A Gaussian is separable: about 30 operations instead of 225 in the direct model."
5. End: "Same separable kernel, less work. Full C++ lesson in the course repo."

## References / attribution

- OpenCV documentation for `filter2D`, `boxFilter`, `sepFilter2D`, `GaussianBlur`, and `getGaussianKernel`
- Standard Gaussian-function and linear-systems definitions
- Repository source code and generated lesson outputs
