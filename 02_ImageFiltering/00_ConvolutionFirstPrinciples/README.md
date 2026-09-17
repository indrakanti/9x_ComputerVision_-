# Convolution from First Principles

This lesson explains the operation at the center of classical image filtering and modern CNNs: sliding a kernel across an image and computing a weighted sum.

## Learning objectives

By the end of this lesson you should be able to:

- write the 2-D filtering equation from memory
- explain the difference between correlation and convolution
- implement the operation manually in C++
- explain what border handling changes at the edges
- compare a manual implementation with OpenCV `filter2D()`
- reason about the runtime cost of an `M x N` kernel on an image

## Correlation vs convolution

For an image `I` and kernel `K`, correlation can be written as:

\[
G(x,y)=\sum_{i=-a}^{a}\sum_{j=-b}^{b} K(i,j) I(x+j,y+i)
\]

True convolution flips the kernel in both dimensions first:

\[
G(x,y)=\sum_{i=-a}^{a}\sum_{j=-b}^{b} K(i,j) I(x-j,y-i)
\]

For symmetric kernels such as a box blur or many Gaussian kernels, the flip produces the same kernel. That can hide the distinction.

**Important OpenCV detail:** `cv::filter2D()` performs correlation. To perform true convolution with `filter2D()`, flip the kernel first.

## What the C++ example does

`convolution.cpp` implements the sliding-window operation manually for grayscale `CV_8UC1` input and produces a floating-point result. It can then compare the manual result against OpenCV.

Supported kernels:

- `box` — 3x3 averaging kernel
- `sharpen` — center-positive sharpening kernel
- `edge-x` — asymmetric horizontal-gradient kernel useful for showing the difference between correlation and convolution

Supported border modes:

- `constant` — pixels outside the image are zero
- `replicate` — copy the closest edge pixel outward

## Build

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_convolution
```

## Deterministic self-test

```bash
./build/cv9x_convolution --self-test
```

The test uses an asymmetric kernel and checks both correlation and convolution under both supported border modes against OpenCV.

You can also run all registered course tests:

```bash
ctest --test-dir build --output-on-failure
```

## Run on the course image

```bash
./build/cv9x_convolution \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/convolution_box \
  --kernel box \
  --operation correlation \
  --border replicate
```

Try a sharpening kernel:

```bash
./build/cv9x_convolution \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/convolution_sharpen \
  --kernel sharpen
```

To make the correlation/convolution difference visible:

```bash
./build/cv9x_convolution \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/correlation_edge \
  --kernel edge-x \
  --operation correlation

./build/cv9x_convolution \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/convolution_edge \
  --kernel edge-x \
  --operation convolution
```

## Outputs

The program writes:

- `manual.png`
- `opencv_reference.png`
- `difference.png`

It also prints the maximum absolute numerical difference between the manual implementation and OpenCV.

## Border handling matters

At an interior pixel, the complete kernel is inside the image. At the border, part of the kernel would extend outside the image. A real implementation needs a policy.

`BORDER_CONSTANT` effectively imagines zeros beyond the image boundary. `BORDER_REPLICATE` extends the nearest edge pixel. Different policies produce different values near the image boundary even when the kernel is identical.

This becomes important in production pipelines because repeated filtering can make border artifacts visible or influence downstream feature extraction.

## Runtime cost

For an image of width `W`, height `H`, and a `Kx x Ky` kernel, a direct implementation performs approximately:

\[
O(W H K_x K_y)
\]

multiply-accumulate work.

For small kernels this is often acceptable. Larger separable kernels can frequently be decomposed into one horizontal and one vertical pass, reducing the cost substantially. Gaussian filtering is the next natural example.

## Engineering notes

The manual implementation is intentionally straightforward rather than optimized. Production libraries may use vectorization, cache-aware loops, separable filters, specialized fixed-size kernels, threading, GPU execution, or accelerator-specific implementations.

The lesson keeps the working image in floating point while filtering. That avoids prematurely clipping negative gradient values or sharpened values above 255. Conversion to 8-bit happens only for visualization.

## Common mistakes

- calling `filter2D()` "convolution" without discussing the missing kernel flip
- comparing 8-bit outputs after saturation instead of comparing floating-point results
- forgetting border behavior when validating an implementation
- testing only symmetric kernels, which cannot reveal correlation-vs-convolution mistakes
- normalizing an image before numerical comparison and thereby hiding errors

## Exercises

1. Add a 5x5 box kernel and compare runtime with the 3x3 kernel.
2. Add `reflect` border handling and compare edge pixels.
3. Add a deliberately asymmetric custom kernel and verify that correlation and convolution differ.
4. Implement separable filtering for a Gaussian-like kernel.
5. Measure latency for 640x480, 1280x720, and 1920x1080 inputs.

## Next lesson

The next lesson uses convolution to explain **box filtering vs Gaussian filtering**, including why Gaussian kernels are separable and why that matters for performance.
