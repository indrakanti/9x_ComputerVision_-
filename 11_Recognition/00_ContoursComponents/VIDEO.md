# Video: Contours, Connected Components & Shape Descriptors in C++

YouTube: TBD

Suggested title:

**Contours & Connected Components in C++ — Shape Descriptors Explained**

Suggested thumbnail:

**FROM PIXELS → OBJECT SHAPES**

Target duration: **28–34 minutes**

## Learning objectives

Viewers should understand:

- component labels vs contour boundaries
- contour hierarchy and holes
- area, perimeter, centroid, bounding boxes
- circularity, aspect ratio, extent, solidity
- polygon approximation
- Hu moments
- simple descriptor-based recognition

## Chapters

~~~text
00:00 From motion to object regions
02:00 Binary masks
04:00 Connected components
07:00 4-connectivity vs 8-connectivity
09:00 Region area, centroid and bounding box
12:00 Contours and boundary representation
15:00 Contour hierarchy and holes
18:00 Circularity, extent and solidity
21:00 Polygon approximation
24:00 Hu moments
27:00 Synthetic shape demo
30:00 Descriptor CSV and engineering notes
32:00 Next: HOG + SVM detection
~~~

## Script outline

### 00:00 — Recognition starts with regions

Show the synthetic scene.

Ask:

> If these are foreground pixels, how do we turn them into objects and measurements?

### 02:00 — Binary segmentation

Show the 0/255 mask.

Explain that all later measurements inherit segmentation quality.

### 04:00 — Connected components

Color every connected foreground object differently.

Explain label 0 as background.

Show stats:

- area
- bounding box
- centroid

### 07:00 — Connectivity

Draw a diagonal two-pixel example.

Compare 4- vs 8-connectivity.

### 09:00 — Region statistics

Walk through connectedComponentsWithStats() outputs.

Explain pixel area vs geometric contour area.

### 12:00 — Contours

Trace the circle boundary.

Explain ordered boundary points and CHAIN_APPROX_SIMPLE.

### 15:00 — Holes

Highlight the donut.

Say:

> Connected components says one foreground object. Contour hierarchy says one outer boundary plus one hole.

Show parent/child hierarchy.

### 18:00 — Shape descriptors

Write:

$$
circularity = 4\pi A/P^2
$$

Then cover:

- aspect ratio
- extent
- convex hull
- solidity

### 21:00 — Polygon approximation

Show the triangle/rectangle contours before and after Douglas-Peucker.

Explain epsilon as a fraction of perimeter.

### 24:00 — Hu moments

Explain moment invariants conceptually.

Translate the same rectangle and show the Hu vector remains essentially unchanged.

### 27:00 — Demo

Run:

~~~bash
./build/cv9x_shape_analysis \
  --output-dir build/shape_analysis
~~~

Show all four outputs.

### 30:00 — Engineering notes

Open shape_descriptors.csv.

Discuss what a downstream module actually needs.

Avoid shipping full contours when a compact region record is enough.

### 32:00 — Next

Preview HOG:

> Today we recognized explicit geometric shape. Next we describe local edge orientation over a window and train a classifier.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_shape_analysis

./build/cv9x_shape_analysis \
  --output-dir build/shape_analysis

./build/cv9x_shape_analysis --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**Connected Components and Contours Are NOT the Same Thing**

30–45 seconds:

1. show a donut binary mask
2. connected components: one foreground object
3. contours: outer boundary + inner hole
4. close with: "regions answer membership; contours answer boundary and topology"

## YouTube description

This lesson begins classical recognition by turning binary pixels into labeled objects and measurable shapes.

We compare connected components with contours, inspect contour hierarchy and holes, compute area/perimeter/centroids/bounding boxes, derive circularity/extent/solidity, simplify contours into polygons, and use Hu moments for invariant shape description.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/11_Recognition/00_ContoursComponents

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #Contours #ConnectedComponents #ShapeAnalysis #Linux
