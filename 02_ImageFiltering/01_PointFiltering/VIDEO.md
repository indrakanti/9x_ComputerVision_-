# Video — Image Brightness, Contrast, Gamma & Thresholding from First Principles

YouTube: **TBD**

Suggested title:

> **Image Brightness, Contrast, Gamma & Thresholding from First Principles | OpenCV C++**

Suggested thumbnail text:

> **PIXEL MATH → IMAGE**

Target duration: **16–22 minutes**

## Learning objectives

By the end of the video, the viewer should be able to:

- describe point processing as `output = f(input)`
- derive brightness, contrast, gamma, inversion, and threshold operations
- explain why 8-bit arithmetic saturates at 0 and 255
- implement gamma correction efficiently using a LUT
- build and run the C++ example on Linux
- verify the result with deterministic pixel-level tests

## Visual hook

Open with the synthetic image and show six outputs side by side:

1. brighter
2. darker
3. increased contrast
4. gamma corrected
5. inverted
6. thresholded

Narration idea:

> Every one of these images can be produced without looking at a single neighboring pixel. The entire operation is just a function applied independently to each intensity value. That simple idea is the starting point for understanding much more complex image-processing pipelines.

## Chapter plan

```text
00:00 What one pixel can teach us about computer vision
01:10 Point operations: output = f(input)
02:30 Brightness and 8-bit saturation
05:00 Contrast scaling
07:00 Gamma correction and lookup tables
10:30 Image inversion
11:30 Binary thresholding and the exact boundary condition
13:30 C++ implementation and CLI
16:30 Deterministic self-test
18:30 Engineering takeaways and exercise
```

## Script outline

### 1. Point operations

Draw one input pixel entering a function block and one output pixel leaving it.

Show:

\[
I_{out} = f(I_{in})
\]

Explain that no neighborhood is required. Contrast this briefly with the next major course topic, convolution, where neighboring pixels matter.

### 2. Brightness

Show:

\[
I_{out} = \operatorname{sat}(I_{in}+\beta)
\]

Use two numerical examples:

```text
20 + 50 = 70
240 + 50 = 290 -> 255 after saturation
```

Engineering point: integer range and clipping are part of the observable algorithm behavior.

### 3. Contrast

Show:

\[
I_{out} = \operatorname{sat}(\alpha I_{in})
\]

Use `alpha = 1.5` and explain why high values clip. Mention that this introductory transform scales around zero; later pipelines may scale around a midpoint or operate on luminance.

### 4. Gamma

Show normalized intensity first:

\[
x=\frac{I_{in}}{255}
\]

Then:

\[
I_{out}=255x^{\gamma}
\]

Plot or sketch curves for:

- `gamma = 0.5`
- `gamma = 1.0`
- `gamma = 2.0`

Explain why the code creates a 256-entry LUT: an 8-bit input can only have 256 different values, so the nonlinear calculation can be performed once per possible value rather than once per pixel.

### 5. Inversion

Show:

\[
I_{out}=255-I_{in}
\]

Use `64 -> 191` as the numerical example.

### 6. Thresholding

Show the actual behavior used by OpenCV `THRESH_BINARY`:

\[
I_{out}=
\begin{cases}
0, & I_{in}\le T \\
255, & I_{in}>T
\end{cases}
\]

Call out the boundary explicitly: **a value equal to the threshold becomes zero**.

### 7. C++ walkthrough

Focus the code walkthrough on these functions:

```text
adjustBrightness()
adjustContrast()
gammaCorrection()
invertImage()
thresholdImage()
```

Then show the CLI and explain why input/output paths are passed explicitly instead of being hidden in relative working-directory assumptions.

## Commands demonstrated

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_point_filter
```

Run the lesson:

```bash
./build/cv9x_point_filter \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/point_filter_results
```

Show parameter exploration:

```bash
./build/cv9x_point_filter \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/point_filter_gamma2 \
  --gamma 2.0
```

Run validation:

```bash
./build/cv9x_point_filter --self-test
```

and then:

```bash
ctest --test-dir build --output-on-failure
```

## Expected output

The normal run should create:

```text
brightness_adjustment.png
contrast_adjustment.png
darkening.png
gamma_correction.png
image_inversion.png
thresholding.png
```

The self-test should print:

```text
Point-filter self-test passed.
```

## Engineering takeaway

End with three points:

1. **Pixel format matters.** `uint8` arithmetic has a finite range.
2. **The equation matters.** "Gamma correction" is ambiguous unless the exact exponent convention is stated.
3. **Visual inspection is not a test.** Known pixel inputs give deterministic expected outputs and belong in CI.

## Viewer exercise

Implement midpoint contrast:

\[
I_{out}=\operatorname{sat}(\alpha(I_{in}-m)+m)
\]

Use `m = 128`, add an exact self-test, and compare its output with contrast scaling around zero.

## Short / clip idea

**Title:** Why does `240 + 50` become `255` in an image?

30–45 seconds showing 8-bit saturation versus arithmetic overflow/wraparound, ending with the full-video pointer.

## YouTube description

```text
In this lesson we build the first complete image-processing pipeline in the 9x Computer Vision series. We derive brightness, contrast, gamma correction, inversion, and thresholding from pixel-level math, implement them in C++ with OpenCV on Linux, and verify them with deterministic tests.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/02_ImageFiltering/01_PointFiltering

Full course repository:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- point operations and 8-bit saturation
- brightness and contrast transforms
- gamma correction using a lookup table
- binary threshold boundary behavior
- reproducible C++ CLI design
- deterministic computer-vision testing

#ComputerVision #OpenCV #CPP #Linux #EmbeddedAI
```

## Recording checklist

- [ ] screen capture at 1080p or higher
- [ ] terminal font large enough for mobile viewers
- [ ] show equations before code
- [ ] show original and outputs side by side
- [ ] demonstrate at least two gamma values
- [ ] run `--self-test` on screen
- [ ] add final YouTube URL to this file and `VIDEO_SERIES.md`
