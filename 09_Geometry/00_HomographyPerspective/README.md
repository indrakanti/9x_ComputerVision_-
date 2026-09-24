# Module 10/11 — Homography & Perspective Transforms from First Principles

This lesson connects feature matching to projective geometry.

In the previous lesson, RANSAC estimated a homography from feature correspondences. Here we slow down and answer:

> What exactly is a homography, how do we solve it, and how does it turn a perspective view into a rectified image?

The implementation manually solves the 8 independent homography parameters from four point correspondences and performs inverse perspective warping with bilinear interpolation.

OpenCV \`warpPerspective()\` is used only as a reference.

## Learning objectives

By the end of the lesson you should be able to:

- explain homogeneous image coordinates
- explain why a 3x3 homography has 8 independent degrees of freedom
- derive the two linear equations contributed by one point correspondence
- solve a four-point homography as an 8x8 linear system
- project points through a homography
- explain why image warping should normally use inverse mapping
- implement bilinear sampling
- rectify a perspective quadrilateral into a front-facing rectangle
- identify degenerate point configurations
- explain where homography assumptions stop being valid

## 1. Projective coordinates

A 2-D image point:

$$
(x,y)
$$

is represented homogeneously as:

$$
\mathbf{x} =
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

A homography maps:

$$
\mathbf{x}' \sim H\mathbf{x}
$$

where:

$$
H =
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
$$

The symbol \(\sim\) means equality up to nonzero scale.

## 2. Why only 8 degrees of freedom?

Multiplying every element of \(H\) by the same nonzero constant does not change the final projected point after homogeneous normalization.

So one matrix scale is arbitrary.

We can choose:

$$
h_{33}=1
$$

for the four-point teaching solve.

That leaves eight unknowns.

## 3. Point mapping

For source point \((x,y)\):

$$
u =
\frac{
h_{11}x+h_{12}y+h_{13}
}{
h_{31}x+h_{32}y+h_{33}
}
$$

$$
v =
\frac{
h_{21}x+h_{22}y+h_{23}
}{
h_{31}x+h_{32}y+h_{33}
}
$$

The denominator is the projective part.

If it approaches zero, the mapped point approaches infinity and the transform becomes numerically problematic in that region.

## 4. Two equations from one correspondence

Cross-multiplying gives linear equations in the unknown homography coefficients.

For \(u\):

$$
x h_{11}
+y h_{12}
+h_{13}
-u x h_{31}
-u y h_{32}
=
u
$$

For \(v\):

$$
x h_{21}
+y h_{22}
+h_{23}
-v x h_{31}
-v y h_{32}
=
v
$$

Each correspondence gives two equations.

Four point pairs give:

$$
4\times2=8
$$

equations for the eight independent unknowns.

## 5. Manual 8x8 solve

The implementation creates:

$$
A\mathbf{h}=\mathbf{b}
$$

where:

$$
\mathbf{h}
=
[h_{11},h_{12},h_{13},h_{21},h_{22},h_{23},h_{31},h_{32}]^T
$$

and solves the linear system with a generic linear solver.

That is different from calling a homography convenience API: the actual projective equations are visible in the source.

## 6. Point ordering

The CLI expects source points in this order:

~~~text
0 = top-left
1 = top-right
2 = bottom-right
3 = bottom-left
~~~

The destination is the corresponding rectangle.

Wrong ordering can produce a reflected, twisted, or otherwise incorrect warp even when the math itself is valid.

## 7. Rectification target

If output width/height are not specified, the lesson estimates them from opposite source edges:

~~~text
width  = max(top edge, bottom edge)
height = max(left edge, right edge)
~~~

You can override them using:

~~~text
--width
--height
~~~

## 8. Why inverse mapping?

A naive forward warp does:

~~~text
source pixel -> destination location
~~~

But destination coordinates rarely land exactly on integer pixels.

This creates holes.

Instead, for every destination pixel, inverse mapping asks:

> Which source coordinate should provide this output pixel?

So:

$$
\mathbf{x}_{src}
\sim
H^{-1}\mathbf{x}_{dst}
$$

This guarantees every destination pixel gets one sampling attempt.

## 9. Bilinear interpolation

The inverse-mapped source coordinate is usually fractional:

~~~text
x = 53.27
y = 41.82
~~~

The implementation interpolates the four neighboring source pixels.

If:

$$
f_x=x-\lfloor x \rfloor
$$

and

$$
f_y=y-\lfloor y \rfloor
$$

then the value is blended first horizontally and then vertically.

This produces smoother results than nearest-neighbor sampling.

## 10. Manual vs OpenCV result

The executable writes:

~~~text
02_manual_rectified.png
03_opencv_rectified_reference.png
04_manual_vs_opencv_difference.png
~~~

Small differences are expected because interpolation implementations may quantize or round differently.

The comparison is a reference—not a requirement for bit-exact equivalence.

## 11. Degenerate geometry

A homography cannot be reliably estimated from arbitrary four points.

Examples of poor/invalid configurations include:

- all source points collinear
- all destination points collinear
- duplicate points
- nearly collinear points
- extremely small quadrilateral area
- bad correspondence ordering

The teaching implementation explicitly rejects zero/near-zero quadrilateral area before solving.

## 12. Homography is not general 3-D geometry

A single homography is appropriate for:

- approximately planar scenes
- plane-to-plane mappings
- pure camera rotation
- document/poster/marker rectification
- some panorama conditions

It is generally not enough for a non-planar scene under translation because different depths produce parallax.

That transition leads later to:

- fundamental matrix
- essential matrix
- stereo geometry
- visual odometry
- SLAM

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_homography_perspective
~~~

## Run: document or planar-object rectification

Example:

~~~bash
./build/cv9x_homography_perspective \
  --input path/to/document_photo.png \
  --output-dir build/homography \
  --src 122 84 518 110 560 690 88 650
~~~

Optional explicit output size:

~~~bash
--width 600 --height 800
~~~

## Outputs

~~~text
01_source_quadrilateral.png
02_manual_rectified.png
03_opencv_rectified_reference.png
04_manual_vs_opencv_difference.png
~~~

## Deterministic self-test

~~~bash
./build/cv9x_homography_perspective --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- identity homography solve
- identity point projection
- recovery of a known perspective transform
- correct projection of unseen probe points
- homography inverse round-trip
- rejection of a collinear/degenerate quadrilateral
- exact identity behavior of the manual inverse image warp

## Common failure cases

### Point order is inconsistent

The source and destination point order must represent the same physical corners.

### Forward warping leaves holes

Use inverse mapping.

### Using nearest neighbor for rectification

It works, but produces visible aliasing and blockiness.

### Treating every four-point solve as trustworthy

Check point geometry and numeric conditioning.

### Applying one homography across strong parallax

A single plane model cannot explain arbitrary 3-D depth variation.

## Engineering notes

A production projective-warp API should make these contracts explicit:

~~~text
coordinate convention
source/destination ordering
pixel-center convention
interpolation mode
border mode
numeric precision
validity / degeneracy status
homography direction
output size
error metrics
~~~

The same habits become important later in camera calibration, stereo, optical flow, visual odometry, and tracking.

## How this connects to live tracking

Homography is not only for document scanning.

Across video frames, feature correspondences can estimate frame-to-frame planar motion.

That enables:

- planar object tracking
- image stabilization
- motion compensation
- panorama/video mosaics
- region tracking
- background alignment

The next stages of the course generalize from one pair of images to a continuous stream of frames.

## Exercises

1. Replace bilinear interpolation with nearest neighbor and compare.
2. Add a color-image manual warp.
3. Add normalized DLT for more than four correspondences.
4. Compute the condition number of the 8x8 system.
5. Track four planar points across video and update the homography every frame.
6. Build a live document/poster tracker using feature matches + RANSAC + homography.
7. Compare manual rectification with OpenCV across several perspective strengths.

## Next lesson

The geometry track continues with camera coordinates, intrinsics, extrinsics, and projection.

The continuous-perception track then turns pairs of images into frame sequences using optical flow and feature tracking.

See [VIDEO.md](VIDEO.md).
