# Video: Sobel & Scharr Image Gradients in C++ — Magnitude, Direction & Signed Derivatives

YouTube: TBD

Suggested title:

**Sobel & Scharr Image Gradients in C++ — Magnitude, Direction & Signed Derivatives**

Suggested thumbnail:

**PIXELS -> DIRECTION**

Target duration: **18–24 minutes**

## Learning objectives

By the end of the video, viewers should understand:

- what an image derivative measures
- how Sobel combines smoothing and differentiation
- why (G_x) and (G_y) are signed
- how magnitude and direction are computed
- why `CV_32F` is preferable to immediate 8-bit conversion
- what Scharr changes relative to 3x3 Sobel
- how to validate the OpenCV operators against explicit kernels

## Visual hook

Start with a bright vertical stripe on a dark background.

Show the signed x-gradient:

- one side bright
- the other side dark
- gray where the derivative is zero

Narration:

> An edge is not just a white line. One side of this stripe has a positive derivative and the other has a negative derivative. If we take the absolute value too early, we throw away that information.

Then show magnitude and direction.

## Chapters

```text
00:00 An edge has a sign
01:20 From pixels to partial derivatives
03:40 Deriving the Sobel kernels
06:20 Why Gx and Gy must stay signed
08:30 Gradient magnitude
10:00 Gradient direction with atan2
12:10 Sobel vs Scharr
14:20 C++ implementation
17:40 Deterministic tests
20:20 Engineering takeaways
22:00 Exercise and next lesson
```

## Script outline

### 00:00 — An edge has a sign

Display:

- original stripe
- signed (G_x)
- absolute (G_x)

Explain that the absolute visualization is easier to view but discards edge polarity.

### 01:20 — From pixels to derivatives

Write:

[

abla I = [G_x, G_y]^T
]

Explain the discrete approximation:

[
[-1,0,1]
]

Use a simple row of pixel values and calculate one derivative manually.

### 03:40 — Deriving Sobel

Show:

[
[-1,0,1]
]

and perpendicular smoothing:

[
[1,2,1]^T
]

Form the outer product to obtain:

[
egin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
end{bmatrix}
]

Then transpose for (G_y).

Connect this directly to the convolution/correlation lesson.

### 06:20 — Signed gradients

Show a rising edge and falling edge.

Explain why `CV_8U` cannot faithfully store both without remapping.

Show the course's signed visualization convention:

```text
negative -> dark
zero     -> gray
positive -> bright
```

Stress:

> That PNG is a debug visualization. The algorithm still owns the floating-point gradient.

### 08:30 — Magnitude

Write:

[
M = sqrt{G_x^2 + G_y^2}
]

Show magnitude for a horizontal, vertical, and diagonal edge.

### 10:00 — Direction

Write:

[
	heta = operatorname{atan2}(G_y,G_x)
]

Use synthetic ramps:

- x ramp -> 0 degrees
- y ramp -> 90 degrees
- x+y ramp -> 45 degrees

Explain angle wraparound.

### 12:10 — Sobel vs Scharr

Show Sobel x:

[
egin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
end{bmatrix}
]

Show Scharr x:

[
egin{bmatrix}
-3&0&3\\
-10&0&10\\
-3&0&3
end{bmatrix}
]

Explain that Scharr is designed for improved rotational accuracy at 3x3.

Caution that raw magnitudes have different scale because kernel gains differ.

### 14:20 — C++ implementation

Walk through:

- CLI parsing
- grayscale -> `CV_32F`
- `computeSobel()`
- `computeScharr()`
- `cv::magnitude()`
- `cv::phase()`
- separate visualization helpers
- explicit border selection

Point out the deliberate absence of `convertScaleAbs()` in the computational path.

### 17:40 — Deterministic tests

Run:

```bash
./build/cv9x_gradients --self-test
```

Explain each test:

1. constant field -> zero gradient
2. x ramp -> 0-degree direction
3. y ramp -> 90-degree direction
4. diagonal ramp -> 45 degrees
5. explicit Sobel kernel == `cv::Sobel()`
6. explicit Scharr kernel == `cv::Scharr()`
7. bright stripe -> positive and negative gradient polarity

Then run:

```bash
ctest --test-dir build --output-on-failure
```

### 20:20 — Engineering takeaways

Three points:

1. **Derivative sign is information.**
2. **Visualization format is not algorithm format.**
3. **Noise and border policy affect derivatives strongly.**

Connect to production perception:

> Once gradients feed Canny, HOG, SIFT, optical flow, or another downstream stage, throwing away sign or precision becomes an interface error, not just a visualization choice.

### 22:00 — Exercise and next lesson

Exercise:

Add Prewitt kernels and compare their directional response against Sobel and Scharr.

Next:

**Image derivatives and edge detection**, leading to Canny.

## Commands demonstrated

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_gradients
```

Run:

```bash
./build/cv9x_gradients \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/gradient_results \
  --method both \
  --border reflect101
```

Test:

```bash
./build/cv9x_gradients --self-test
ctest --test-dir build --output-on-failure
```

## Expected output

```text
sobel_gx_signed.png
sobel_gy_signed.png
sobel_magnitude.png
sobel_direction.png
scharr_gx_signed.png
scharr_gy_signed.png
scharr_magnitude.png
scharr_direction.png
```

## Short / clip idea

Title:

**Why Your Sobel Edge Has TWO Signs**

30–45 second structure:

1. show bright stripe
2. show left edge positive
3. show right edge negative
4. show what absolute value does
5. close with: "Keep signed gradients for the algorithm. Take abs only for display."

## YouTube description

In this lesson we move from image filtering to image derivatives and implement Sobel and Scharr gradients in C++ on Linux.

We derive the kernels, preserve signed Gx/Gy responses, compute gradient magnitude and direction, explain why Scharr exists, and validate OpenCV against explicit 3x3 kernels using deterministic tests.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/03_ImageGradients/00_SobelScharr

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- Sobel x/y derivative kernels
- signed gradient representation
- gradient magnitude and direction
- atan2 and angle interpretation
- Sobel vs Scharr
- border handling
- deterministic gradient tests

#ComputerVision #OpenCV #CPP #Linux #Sobel #Scharr #ImageProcessing

## Recording checklist

- [ ] show signed edge polarity visually
- [ ] derive Sobel from smoothing x differentiation
- [ ] explain `CV_32F`
- [ ] show magnitude formula
- [ ] show direction formula
- [ ] compare Sobel and Scharr kernels
- [ ] run both operators
- [ ] run self-test
- [ ] show explicit-kernel equivalence
- [ ] preview Canny / edge detection
