# Video: Convolution Explained Visually — The Core Operation of Computer Vision

YouTube: TBD

## Target length

18–24 minutes

## Learning objectives

- understand the sliding-window weighted-sum operation
- distinguish correlation from true convolution
- understand why symmetric kernels can hide the difference
- implement filtering manually in C++
- compare against OpenCV `filter2D()`
- understand border handling and direct-convolution cost

## Visual hook

Open with the same image processed by three kernels:

1. box blur
2. sharpen
3. horizontal edge detector

Then show one 3x3 kernel physically moving across a magnified pixel grid. The message is: **the same operation can blur, sharpen, or detect edges — only the weights change.**

## Suggested chapters

```text
00:00 Why convolution matters
01:15 Image + kernel intuition
03:30 One output pixel by hand
06:00 Correlation vs convolution
08:30 OpenCV filter2D() detail
10:15 Manual C++ implementation
14:30 Border handling
17:00 Compare manual vs OpenCV
19:30 Runtime and production considerations
22:00 Exercises and next lesson
```

## Script outline

### 1. Why this operation matters

Start with three outputs from the same input image. Explain that each result comes from the same sliding-window machinery and a different kernel.

Connect this to later topics: Gaussian blur, Sobel gradients, Laplacian filters, feature extraction, and CNN convolution layers.

### 2. Build one output pixel manually

Draw a 3x3 image patch and a 3x3 kernel. Multiply corresponding values and sum them.

Write:

\[
G(x,y)=\sum_i\sum_j K(i,j)I(x+j,y+i)
\]

Emphasize that every output pixel repeats this local weighted sum.

### 3. Correlation vs convolution

Show an asymmetric kernel, then rotate it 180 degrees.

Explain:

- correlation uses the kernel as written
- convolution flips the kernel horizontally and vertically before applying it

Use the `edge-x` kernel because the sign/orientation difference is visually obvious.

Then state the practical OpenCV point:

> `cv::filter2D()` performs correlation. For mathematical convolution, flip the kernel first.

Mention that symmetric kernels such as the box kernel produce the same result either way, which is why the distinction is easy to miss.

### 4. Walk through the C++ implementation

Show `manualCorrelation()`.

Highlight:

- explicit padding
- nested image loops
- nested kernel loops
- floating-point accumulation
- output stored as `CV_32F`

Then show `manualFilter()` and the kernel flip for convolution.

### 5. Border behavior

At the upper-left pixel, part of the kernel extends outside the image.

Compare:

- constant zero padding
- replicate padding

Run the same kernel with both modes and point out that differences are concentrated around image boundaries.

### 6. Numerical validation

Show the command:

```bash
./build/cv9x_convolution --self-test
```

Explain why the self-test uses an **asymmetric** kernel. A symmetric kernel would not catch a missing flip.

Then run a normal image and show:

```text
Manual vs OpenCV max absolute error: 0
```

or the measured floating-point tolerance.

### 7. Runtime engineering note

Write:

\[
O(W H K_x K_y)
\]

Explain that direct filtering scales with image size and kernel area.

Preview the next lesson: Gaussian filtering can exploit separability so a 2-D kernel becomes two 1-D passes.

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_convolution
./build/cv9x_convolution --self-test
```

```bash
./build/cv9x_convolution \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/convolution_box \
  --kernel box \
  --operation correlation \
  --border replicate
```

```bash
./build/cv9x_convolution \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/convolution_edge \
  --kernel edge-x \
  --operation convolution
```

## Files to show on screen

- `convolution.cpp`
- `README.md`
- generated `manual.png`
- generated `opencv_reference.png`
- generated `difference.png`

## Engineering takeaways

- keep intermediate filter results in a type that can represent negative and >255 values
- validate algorithms before visualization conversion
- make border behavior explicit
- use asymmetric test kernels when testing kernel orientation
- optimized production filtering may look very different internally from the pedagogical nested loops

## Exercise for viewers

Add `reflect` border handling and a 5x5 averaging kernel, then compare both output and runtime.

## Suggested title

`Convolution Explained Visually in C++ — Build It from Scratch + OpenCV Comparison`

Alternative search-oriented title:

`Image Convolution from Scratch in C++ — Kernels, Borders & OpenCV filter2D`

## Thumbnail concept

Large 3x3 kernel in the center, an image patch on the left, filtered image on the right. Minimal text:

**CONVOLUTION**

Small secondary text: `C++ FROM SCRATCH`

## YouTube description draft

In this lesson we build image convolution from first principles in C++ on Linux. We manually slide a kernel over an image, compare the result with OpenCV, examine border handling, and explain the often-missed difference between correlation and true convolution.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/02_ImageFiltering/00_ConvolutionFirstPrinciples

Full course repository:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- how a 2-D image kernel actually works
- correlation vs convolution
- why OpenCV filter2D uses correlation
- constant vs replicate borders
- how to validate a manual implementation
- direct filtering runtime cost

#ComputerVision #OpenCV #CPP #Linux #ImageProcessing

## Short / clip idea

**Hook:** "OpenCV filter2D is called a filter, but is it actually doing mathematical convolution?"

In 30–45 seconds, show an asymmetric kernel, flip it, and demonstrate why correlation and convolution differ while a symmetric blur kernel hides the distinction.
