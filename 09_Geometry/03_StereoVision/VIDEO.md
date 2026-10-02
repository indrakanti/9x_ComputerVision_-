# Video: Stereo Vision & Epipolar Geometry — From Two Cameras to Depth

YouTube: TBD

Suggested title:

**Stereo Vision & Epipolar Geometry — From Two Cameras to Metric Depth in C++**

Suggested thumbnail:

**TWO CAMERAS → DEPTH**

Target duration: **32–38 minutes**

## Learning objectives

Viewers should understand:

- relative stereo pose
- essential and fundamental matrices
- the epipolar constraint
- epipolar lines
- why rectification matters
- disparity
- the equation Z = fB/d
- sparse triangulation
- dense SGBM depth
- the engineering assumptions required for metric stereo

## Chapters

~~~text
00:00 Why two cameras give depth
02:00 Stereo coordinate convention
05:00 Projection in the left and right cameras
08:00 Derive disparity d = fB/Z
11:00 Essential matrix E
14:00 Fundamental matrix F
17:00 Epipolar line intuition
20:00 Stereo rectification
23:00 Sparse triangulation
26:00 Dense disparity with SGBM
30:00 Convert disparity to metric depth
33:00 Error grows with distance
35:00 Deterministic tests
37:00 Next: optical flow and motion
~~~

## Script outline

### 00:00 — Why a second camera helps

Show one image of a point.

Ask:

> How far away is this point?

Then show the same point in two cameras.

Introduce:

~~~text
known camera separation
+
image displacement
=
depth information
~~~

### 02:00 — Coordinate convention

Use the left camera as the reference.

Write:

$$
P_R = R P_L + t
$$

For the ideal rectified rig:

$$
R=I
$$

and:

$$
t =
\begin{bmatrix}
-B\\
0\\
0
\end{bmatrix}
$$

Clarify the difference between translation-vector sign and positive physical baseline magnitude.

### 05:00 — Two projections

Write:

$$
u_L = fX/Z+c_x
$$

and:

$$
u_R = f(X-B)/Z+c_x
$$

### 08:00 — Derive disparity

Subtract:

$$
d=u_L-u_R=fB/Z
$$

Then:

$$
Z=fB/d
$$

Show:

~~~text
near -> large d
far  -> small d
~~~

### 11:00 — Essential matrix

Write:

$$
E=[t]_\times R
$$

Explain:

> E is the relative two-camera geometry expressed in normalized camera coordinates.

### 14:00 — Fundamental matrix

Write:

$$
F=K^{-T}EK^{-1}
$$

Then:

$$
x_R^T F x_L=0
$$

Explain that F operates directly on pixel coordinates.

### 17:00 — Epipolar line

Select one left-image point.

Compute:

$$
l_R=Fx_L
$$

Draw the line in the right image.

Explain why the correct point must lie on this line in the ideal model.

### 20:00 — Rectification

Show two unrectified camera views where epipolar lines tilt.

Then transform them so corresponding points lie on common rows.

Explain:

> Rectification turns correspondence from a general line search into a horizontal search.

Show the deterministic y-alignment test.

### 23:00 — Sparse triangulation

Take one known correspondence.

Use P1 and P2.

Recover the original 3-D point.

Connect this to future:

- visual odometry
- structure from motion
- SLAM

### 26:00 — Dense SGBM

Use the deterministic synthetic stereo pair.

Show the known texture shift.

Run:

~~~bash
./build/cv9x_stereo_vision \
  --output-dir build/stereo \
  --fx 800 \
  --baseline 0.20 \
  --depth 5
~~~

Reveal the known 32-pixel disparity.

### 30:00 — Metric depth

Show SGBM's fixed-point representation.

Explain the divide-by-16 conversion.

Then apply:

$$
Z=fB/d
$$

Clarify units:

~~~text
pixels * meters / pixels = meters
~~~

### 33:00 — Stereo depth uncertainty

Write:

$$
\frac{\partial Z}{\partial d}
=
-\frac{fB}{d^2}
$$

Show how the same 0.25-pixel disparity error produces much larger distance error far away.

### 35:00 — Deterministic tests

Run:

~~~bash
./build/cv9x_stereo_vision --self-test
~~~

Walk through:

- epipolar residual
- epipolar point-line distance
- rectified y agreement
- fB/d
- X/Y/Z recovery
- triangulation

Explain why CI does not assert exact SGBM pixels.

### 37:00 — Next

Show two sequential video frames instead of two cameras.

Preview:

> Stereo is correspondence across space. Next we study correspondence across time.

Introduce optical flow.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_stereo_vision

./build/cv9x_stereo_vision \
  --output-dir build/stereo \
  --width 960 --height 540 \
  --fx 800 --fy 800 \
  --baseline 0.20 \
  --depth 5.0 \
  --num-disparities 96 \
  --block-size 5

./build/cv9x_stereo_vision --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**Why Far-Away Stereo Depth Gets Noisy**

30–45 seconds:

1. show Z = fB/d
2. near object has large disparity
3. far object has tiny disparity
4. same 0.25-pixel matching error becomes a much larger depth error
5. close with: "stereo range is geometry, not just algorithm quality"

## YouTube description

This lesson builds stereo depth from geometry.

We derive the essential and fundamental matrices, validate the epipolar constraint, rectify a stereo rig, triangulate sparse points, compute dense disparity using StereoSGBM, and convert disparity into metric depth using the calibrated focal length and baseline.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/09_Geometry/03_StereoVision

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #StereoVision #EpipolarGeometry #Depth #Linux
