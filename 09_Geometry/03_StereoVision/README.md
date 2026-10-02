# Module 14 — Stereo Vision, Epipolar Geometry, Disparity & Depth

A single calibrated camera maps 3-D points into one image.

A stereo pair asks a deeper question:

> Given two calibrated views of the same scene, how do we recover depth?

The complete path is:

~~~text
two calibrated cameras
        |
        v
relative pose R, t
        |
        v
essential matrix E
        |
        v
fundamental matrix F
        |
        v
epipolar constraint
        |
        v
stereo rectification
        |
        v
horizontal correspondence search
        |
        v
disparity
        |
        v
metric depth
~~~

This lesson separates the geometry from the dense matching algorithm so the learner understands what StereoSGBM is allowed to assume.

## Learning objectives

By the end of this lesson you should be able to:

- define the relative pose between left and right cameras
- construct the essential matrix from R and t
- construct the fundamental matrix from intrinsics and E
- explain the epipolar constraint
- compute an epipolar line from a point
- explain why stereo rectification simplifies correspondence search
- understand disparity sign and baseline convention
- derive the rectified stereo depth equation
- reconstruct X, Y, Z from disparity
- triangulate a sparse correspondence
- explain what StereoSGBM estimates
- distinguish invalid disparity from infinite/far depth
- understand why metric depth requires calibrated focal length and baseline

## 1. Stereo coordinate convention

The lesson uses the left camera as the reference frame.

A 3-D point expressed in the left camera frame is transformed into the right camera frame as:

$$
P_R = R_{R\leftarrow L} P_L + t_{R\leftarrow L}
$$

For the ideal rectified rig:

$$
R_{R\leftarrow L}=I
$$

and the right camera center lies a physical distance B to the right of the left camera.

Therefore, under the world-to-camera convention used by the course:

$$
t_{R\leftarrow L}
=
\begin{bmatrix}
-B\\
0\\
0
\end{bmatrix}
$$

This sign is important.

The physical baseline magnitude is:

$$
B = \|C_R-C_L\|
$$

and remains positive.

## 2. Two projection equations

For a rectified pair with shared focal length and principal point:

Left camera:

$$
u_L = f_x \frac{X}{Z}+c_x
$$

Right camera:

$$
u_R = f_x \frac{X-B}{Z}+c_x
$$

Subtract:

$$
u_L-u_R
=
f_x\frac{B}{Z}
$$

Define disparity:

$$
d=u_L-u_R
$$

Then:

$$
d=\frac{f_xB}{Z}
$$

and therefore:

$$
Z=\frac{f_xB}{d}
$$

This is the core metric stereo relationship.

## 3. What disparity means

Disparity is the horizontal displacement of a corresponding scene point between the rectified left and right images.

For the convention above:

~~~text
near object  -> larger disparity
far object   -> smaller disparity
infinite Z   -> disparity approaches zero
~~~

A zero or negative disparity is not accepted by the simple depth function in this lesson.

That does not mean every such pixel is physically invalid in every possible stereo convention. It means it is invalid for the specific rectified convention and matcher configuration used here.

## 4. Essential matrix

The essential matrix describes epipolar geometry in normalized camera coordinates.

Given relative rotation R and translation t:

$$
E=[t]_\times R
$$

where:

$$
[t]_\times
=
\begin{bmatrix}
0&-t_z&t_y\\
t_z&0&-t_x\\
-t_y&t_x&0
\end{bmatrix}
$$

The essential matrix depends on relative camera pose, not focal length or principal point.

## 5. Fundamental matrix

Image pixels are not normalized camera coordinates.

With camera intrinsic matrix K:

$$
F=K^{-T} E K^{-1}
$$

For corresponding pixel points:

$$
x_L=
\begin{bmatrix}
u_L\\
v_L\\
1
\end{bmatrix}
,\qquad
x_R=
\begin{bmatrix}
u_R\\
v_R\\
1
\end{bmatrix}
$$

the epipolar constraint is:

$$
x_R^T F x_L = 0
$$

for an ideal noiseless correspondence.

## 6. Epipolar lines

A point in the left image defines a line in the right image:

$$
l_R=F x_L
$$

where:

$$
l_R=
\begin{bmatrix}
a\\
b\\
c
\end{bmatrix}
$$

represents:

$$
au+bv+c=0
$$

The corresponding right-image point should lie on that line.

Point-to-line distance is:

$$
d_{line}
=
\frac{|au+bv+c|}
{\sqrt{a^2+b^2}}
$$

This gives a useful geometric correspondence diagnostic.

## 7. Why epipolar geometry matters

Without stereo geometry, correspondence is a 2-D search problem.

A point in the left image might match anywhere in the right image.

Epipolar geometry reduces that search to one line.

After rectification, the line becomes approximately horizontal.

Then the correspondence problem becomes primarily a 1-D horizontal search.

That is why dense stereo becomes computationally practical.

## 8. Rectification

Real stereo cameras are rarely mounted with mathematically perfect alignment.

They can differ in:

- rotation
- vertical position
- optical-axis direction
- principal point
- distortion

Rectification computes virtual camera orientations that transform the two images so corresponding epipolar lines become horizontal.

This lesson uses OpenCV stereoRectify() for the rectification transform.

The deterministic test starts with a slightly rotated/translationally offset stereo rig and verifies that corresponding points have nearly equal rectified y coordinates afterward.

## 9. Sparse triangulation

Stereo depth does not require a dense disparity map.

Given one left/right correspondence and the two projection matrices:

$$
P_L,\ P_R
$$

we can solve for the 3-D point by triangulation.

The lesson uses OpenCV triangulatePoints() as a linear-algebra reference and verifies that a known synthetic 3-D point is recovered.

This connects stereo matching to later topics such as:

- structure from motion
- visual odometry
- SLAM

## 10. Dense stereo

Dense stereo tries to estimate disparity for many or most image pixels.

The executable uses:

~~~text
StereoSGBM
~~~

for the dense demonstration.

SGBM is not treated as magic.

Its job begins only after the geometric assumptions are established:

~~~text
calibrated cameras
+
rectified images
+
known disparity direction/range
~~~

## 11. Synthetic rectified pair

If no left/right image files are supplied, the executable generates a deterministic textured stereo pair.

The synthetic scene is a fronto-parallel plane at known depth.

The true disparity is:

$$
d_{truth}
=
\operatorname{round}
\left(
\frac{f_xB}{Z}
\right)
$$

The right image is generated by shifting the texture according to that known disparity.

This gives a reproducible dense-stereo demonstration.

## 12. Real stereo input contract

You may also supply:

~~~text
--left left.png
--right right.png
~~~

These images are assumed to already be:

- synchronized
- calibrated
- undistorted as required
- stereo-rectified
- same resolution
- using the supplied fx and baseline

The lesson does not silently rectify arbitrary real image pairs without their calibration.

That would be physically incorrect.

## 13. SGBM parameters

The main exposed parameters are:

~~~text
--num-disparities
--block-size
~~~

num-disparities must be a positive multiple of 16 for OpenCV SGBM.

A larger disparity range allows closer depths but increases computation.

The block size controls the local matching support.

Too small:

- noisier matching
- weak texture instability

Too large:

- blurred depth boundaries
- loss of thin structures

## 14. Fixed-point SGBM output

OpenCV StereoSGBM returns a signed fixed-point disparity representation.

The executable converts it to floating-point pixels using:

$$
d_{px}
=
\frac{d_{raw}}{16}
$$

Do not use the raw 16x-scaled integer directly in the metric depth equation.

## 15. Metric depth

For every valid positive disparity:

$$
Z=
\frac{f_xB}{d}
$$

Units matter.

If:

~~~text
f_x = pixels
B   = meters
d   = pixels
~~~

then:

~~~text
Z = meters
~~~

If baseline is millimeters, depth comes out in millimeters.

## 16. Recovering X and Y

Once Z is known:

$$
X=
(u-c_x)\frac{Z}{f_x}
$$

$$
Y=
(v-c_y)\frac{Z}{f_y}
$$

So a disparity pixel can be back-projected into a 3-D camera-frame point.

The deterministic self-test verifies this entire reconstruction.

## 17. Error behavior

From:

$$
Z=\frac{fB}{d}
$$

depth sensitivity to disparity is:

$$
\frac{\partial Z}{\partial d}
=
-\frac{fB}{d^2}
$$

Since disparity decreases with distance, a one-pixel disparity error causes increasingly large depth error for far objects.

This is a fundamental stereo limitation.

Increasing baseline or focal length improves depth sensitivity, but changes packaging, field of view, overlap, and close-range behavior.

## 18. Outputs

The executable writes:

~~~text
01_left.png
02_right.png
03_disparity.png
04_depth_inverse_visualization.png
05_epipolar_geometry.png
~~~

The depth visualization is an inverse-style display for teaching: nearer valid depth is brighter.

It is not a calibrated color scale.

## 19. Why dense matcher output is not a CI oracle

Stereo matching is affected by:

- texture
- occlusion
- repeated patterns
- block/window settings
- implementation details
- borders
- subpixel interpolation
- uniqueness checks

Therefore CI validates the stereo geometry deterministically:

- E/F relationships
- epipolar constraint
- rectification
- disparity-depth equation
- 3-D reconstruction
- triangulation

The exact dense SGBM pixel output is demonstrated and measured, not asserted as a brittle bit-exact truth.

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_stereo_vision
~~~

## Run synthetic stereo

~~~bash
./build/cv9x_stereo_vision \
  --output-dir build/stereo \
  --width 960 \
  --height 540 \
  --fx 800 \
  --fy 800 \
  --baseline 0.20 \
  --depth 5.0 \
  --num-disparities 96 \
  --block-size 5
~~~

For these defaults:

$$
d =
\frac{800\times0.20}{5.0}
=
32\ pixels
$$

## Run real rectified images

~~~bash
./build/cv9x_stereo_vision \
  --left left_rectified.png \
  --right right_rectified.png \
  --output-dir build/stereo_real \
  --fx 1210.4 \
  --fy 1208.7 \
  --baseline 0.24 \
  --num-disparities 160 \
  --block-size 5
~~~

The supplied calibration must match the images.

## Deterministic self-test

~~~bash
./build/cv9x_stereo_vision --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- expected rectified disparity equals fB/Z
- corresponding rectified points share the same image row
- disparity-to-depth recovers known metric depth
- disparity back-projection recovers X/Y/Z
- zero disparity is rejected
- a rectified correspondence satisfies the epipolar constraint
- the matching point lies on its epipolar line
- sparse triangulation recovers the original 3-D point
- a general slightly unrectified rig satisfies x_R^T F x_L = 0
- stereoRectify aligns the test correspondences onto common rows

## Common failure cases

### Using unrectified images with horizontal disparity search

The matcher assumes the geometry has already been reduced to horizontal correspondence.

### Wrong baseline sign or units

Use a positive physical baseline magnitude in the depth equation.

Be explicit about the sign of the relative translation vector separately.

### Using raw SGBM fixed-point disparity

Divide the OpenCV output by 16 before using pixel disparity.

### Treating zero disparity as zero depth

It is the opposite direction: as disparity approaches zero, ideal depth approaches infinity.

### Ignoring synchronization

Moving objects require left and right images captured at sufficiently matched times.

### Weak texture

A flat wall can satisfy geometry but still provide little correspondence information.

### Repeated texture

Repeated patterns can create ambiguous disparities.

### Occlusion

Some points visible in one camera are physically not visible in the other.

### Expecting precise long-range depth from a short baseline

Stereo uncertainty grows rapidly as disparity becomes small.

## Engineering notes

A production stereo interface should define:

~~~text
left/right camera identity
timestamp synchronization requirement
intrinsics and distortion model
relative R and t
baseline magnitude + units
rectification matrices/maps
rectified image size
fx used for depth
disparity convention/sign
fixed-point scaling
valid disparity range
invalid disparity code
depth units
matcher configuration/version
quality/confidence output
~~~

Calibration, rectification, disparity, and depth parameters must be version-compatible.

## Exercises

1. Sweep depth from 2 m to 50 m and plot disparity.
2. Add +/-0.25-pixel disparity error and plot depth error.
3. Sweep baseline and compare long-range depth sensitivity.
4. Add a foreground rectangle at a second known depth to the synthetic pair.
5. Compare StereoBM and StereoSGBM.
6. Add left-right consistency checking.
7. Use the Q matrix returned by stereoRectify and compare reprojectImageTo3D() with the manual fB/d result.
8. Visualize epipolar lines before and after rectification.
9. Load the calibration_report.yml from the calibration lesson and derive a stereo configuration contract.

## Next lesson

Episode 19 / PR #21 covers optical flow and motion estimation:

- brightness constancy
- image gradients in time
- Lucas-Kanade
- pyramidal optical flow
- feature motion between frames
- forward/backward consistency
- first transition from static image pairs to temporal video processing

See VIDEO.md.
