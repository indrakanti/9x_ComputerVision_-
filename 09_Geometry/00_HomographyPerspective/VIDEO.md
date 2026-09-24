# Video: Homography from First Principles — Perspective Rectification in C++

YouTube: TBD

Suggested title:

**Homography from First Principles — Perspective Rectification in C++**

Suggested thumbnail:

**8 NUMBERS WARP A PLANE**

Target duration: **28–34 minutes**

## Learning objectives

Viewers should understand:

- homogeneous coordinates
- 8 independent homography parameters
- four-point solve
- projective point mapping
- inverse warping
- bilinear interpolation
- document/planar rectification
- degeneracy and parallax limitations

## Chapters

~~~text
00:00 What RANSAC was estimating
02:00 Projective coordinates
05:00 Why homography has 8 DoF
08:00 Derive equations from one point
12:00 Four points -> 8x8 system
15:00 Project a point through H
18:00 Why forward warping creates holes
20:00 Inverse perspective warp
23:00 Bilinear interpolation
26:00 Document rectification demo
29:00 Deterministic tests
31:00 Degeneracy + parallax
33:00 From images to video tracking
~~~

## Script outline

### 00:00 — Connect to PR #15

Show the homography estimated by RANSAC.

Ask:

> What are those nine matrix numbers actually doing?

### 02:00 — Homogeneous coordinates

Introduce:

$$
[x,y,1]^T
$$

and:

$$
x' \sim Hx
$$

Explain scale equivalence.

### 05:00 — 8 DoF

Show 3x3 = 9 entries.

Then explain matrix scale is arbitrary.

Set:

$$
h_{33}=1
$$

for the teaching solve.

### 08:00 — One correspondence

Derive the u and v equations.

Make the connection:

~~~text
1 match -> 2 equations
4 matches -> 8 equations
~~~

### 12:00 — Build A and b

Show the actual C++ matrix assembly.

Then solve:

$$
Ah=b
$$

### 15:00 — Point projection

Show homogeneous denominator.

Animate perspective compression.

### 18:00 — Forward warp problem

Push a source grid through H.

Show holes in destination pixels.

### 20:00 — Inverse warp

Switch perspective:

~~~text
for each output pixel:
    map through inverse H
    sample source
~~~

### 23:00 — Bilinear interpolation

Show four source pixels and fractional weights.

### 26:00 — Rectification demo

Pick four corners around a document/poster.

Run:

~~~bash
./build/cv9x_homography_perspective \
  --input document.png \
  --output-dir build/homography \
  --src x0 y0 x1 y1 x2 y2 x3 y3
~~~

Show:

- source quadrilateral
- manual rectification
- OpenCV reference
- difference image

### 29:00 — Tests

Run self-test.

Explain known transform recovery and identity warp.

### 31:00 — Degeneracy + parallax

Show four collinear points.

Then show a 3-D scene with foreground/background motion.

Explain why one H cannot fit both under translation.

### 33:00 — From images to video

Transition to the larger course goal:

> We are done thinking only in isolated images. Next we begin maintaining correspondences across time.

Preview:

- optical flow
- KLT tracks
- object boxes over frames
- Kalman filtering
- multi-object tracking
- learned temporal perception

## Short idea

**Why Image Warping Uses the Inverse Transform**

1. forward warp creates holes
2. destination-driven inverse mapping fills every output pixel
3. fractional source positions need interpolation

## YouTube description

This lesson derives homography from first principles and implements perspective rectification manually in C++.

We build the 8x8 linear system from four point correspondences, solve the 8 independent projective parameters, implement point projection, inverse warping, and bilinear sampling, then compare against OpenCV.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/09_Geometry/00_HomographyPerspective

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #Homography #ProjectiveGeometry #Linux
