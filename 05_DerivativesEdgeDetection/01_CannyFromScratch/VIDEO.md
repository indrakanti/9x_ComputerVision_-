# Video: Canny Edge Detection from Scratch in C++ — NMS + Hysteresis Explained

YouTube: TBD

Suggested title:

**Canny Edge Detection from Scratch in C++ — NMS + Hysteresis Explained**

Suggested thumbnail:

**WHY CANNY WORKS**

Target duration: **24–30 minutes**

## Learning objectives

Viewers should understand every Canny stage:

- Gaussian smoothing
- Sobel gradients
- magnitude and direction
- angle quantization
- non-maximum suppression
- double thresholding
- 8-connected hysteresis

The video should make it clear that `cv::Canny()` is a compact API for a multi-stage algorithm.

## Visual hook

Open with four images side by side:

1. original
2. raw gradient magnitude
3. non-maximum suppression
4. final Canny edges

Narration:

> The gradient already knows where intensity changes. But those responses are thick, noisy, and disconnected. Canny works because it does three additional things: it thins the response, classifies confidence, and uses connectivity to decide which weak edges survive.

## Chapters

```text
00:00 Why gradient magnitude is not enough
01:30 The complete Canny pipeline
03:00 Gaussian smoothing
04:30 Sobel magnitude and direction
07:00 Quantizing direction
09:00 Non-maximum suppression
13:00 Double threshold
15:30 Hysteresis as graph traversal
19:00 C++ implementation
23:00 Deterministic tests
25:30 Compare with OpenCV Canny
28:00 Engineering takeaways
29:00 Next lesson
```

## Script outline

### 00:00 — Why gradient magnitude is not enough

Show the Episode 07 thresholded gradient map.

Zoom into an edge and show that several adjacent pixels can all have high magnitude.

Question:

> Which one should represent the edge?

Answer:

> The local maximum along the gradient direction.

### 01:30 — Full pipeline

Show:

```text
Gaussian
 -> Sobel
 -> magnitude/direction
 -> NMS
 -> double threshold
 -> hysteresis
```

Explain that the video will implement each stage independently.

### 03:00 — Gaussian smoothing

Recap why differentiation amplifies noise.

Keep this short because Gaussian filtering was already covered earlier.

### 04:30 — Sobel

Show:

$$
M = \sqrt{G_x^2+G_y^2}
$$

$$
\theta = \operatorname{atan2}(G_y,G_x)
$$

Explain that magnitude gives strength while direction tells NMS where to compare.

### 07:00 — Angle quantization

Draw a center pixel with four possible comparison axes.

Show the angle bins:

```text
0
45
90
135
```

Explain 180-degree periodicity.

### 09:00 — Non-maximum suppression

Draw three samples:

```text
2   5   2
    ^
  keep
```

then:

```text
5   2   1
    ^
 suppress
```

Emphasize that comparison neighbors depend on direction.

Show `04_non_maximum_suppression.png`.

### 13:00 — Double threshold

Show three classes:

```text
below low  -> reject
low..high  -> weak
>= high    -> strong
```

Explain why one threshold is fragile.

Show `05_double_threshold.png`.

### 15:30 — Hysteresis

Draw:

```text
S W W W
        \
         W

          W   <- isolated
```

Explain BFS / queue traversal.

Connected weak pixels survive.

Isolated weak pixels disappear.

### 19:00 — C++ implementation

Walk through only the important functions:

- `quantizeDirection()`
- `nonMaximumSuppression()`
- `doubleThreshold()`
- `hysteresis()`
- `runCannyFromScratch()`

Point out that each stage has a clean input/output contract.

### 23:00 — Deterministic tests

Run:

```bash
./build/cv9x_canny --self-test
```

Explain tests for:

- angle bins
- NMS
- threshold classes
- connected weak chain
- isolated weak rejection
- constant image
- vertical step

Then:

```bash
ctest --test-dir build --output-on-failure
```

### 25:30 — Compare with OpenCV

Show:

```text
06_hysteresis_edges.png
07_opencv_canny_reference.png
08_reference_difference.png
```

Print IoU.

Explain why exact equality is not the goal.

### 28:00 — Engineering takeaways

1. Canny is a pipeline, not a single convolution.
2. NMS is directional selection.
3. Hysteresis is graph connectivity.
4. Intermediate representations make debugging and testing much easier.
5. Thresholds only make sense with a defined preprocessing and gradient scale.

### 29:00 — Next lesson

Move from edges to **features/corners**:

- Harris
- Shi-Tomasi
- FAST

## Commands demonstrated

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_canny

./build/cv9x_canny \
  --input 02_ImageFiltering/01_PointFiltering/01_Images/SyntheticImage_Base.png \
  --output-dir build/canny_results \
  --sigma 1.0 \
  --low-threshold 50 \
  --high-threshold 100

./build/cv9x_canny --self-test
ctest --test-dir build --output-on-failure
```

## Short idea

**Canny Edge Detection Is Actually Graph Traversal**

30–45 seconds:

1. show strong and weak pixels
2. connect weak pixels to a strong edge
3. show an isolated weak pixel
4. run hysteresis
5. reveal connected chain survives / isolated pixel disappears

## YouTube description

In this lesson we implement Canny edge detection from scratch in C++ on Linux.

Instead of treating Canny as one OpenCV call, we build and visualize every stage: Gaussian smoothing, Sobel gradients, magnitude/direction, non-maximum suppression, double thresholding, and 8-connected hysteresis.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/05_DerivativesEdgeDetection/01_CannyFromScratch

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

What you will learn:
- Canny pipeline architecture
- direction quantization
- non-maximum suppression
- double thresholding
- hysteresis / graph traversal
- deterministic C++ testing
- comparison with OpenCV Canny

#ComputerVision #OpenCV #CPP #Canny #EdgeDetection #Linux

## Recording checklist

- [ ] show gradient vs NMS vs final edges
- [ ] draw full pipeline
- [ ] explain angle quantization
- [ ] animate NMS neighbor comparison
- [ ] explain weak/strong thresholds
- [ ] animate hysteresis BFS
- [ ] walk through core C++ functions
- [ ] run deterministic test
- [ ] show OpenCV comparison + difference
- [ ] preview corner detection
