# Module 13 — Camera Calibration & Reprojection Error

The previous lesson assumed that the camera matrix K, lens distortion, and camera pose were already known.

Calibration answers:

> How do we estimate those parameters from observations of a target whose geometry we know?

This lesson uses a known planar calibration board, generates many views, estimates the camera model with OpenCV calibrateCamera(), and then independently recomputes every reprojection residual.

The important lesson is not merely how to call the API. It is how to judge whether the result is believable.

## Learning objectives

By the end of this lesson you should be able to:

- explain what camera calibration estimates
- distinguish intrinsic parameters from per-view extrinsics
- explain why known target geometry is required
- explain why one front-facing board image is insufficient
- understand the role of multiple tilted/translated views
- understand planar calibration intuition
- define reprojection residual and reprojection RMSE
- compute per-view and global reprojection error
- distinguish optimizer RMS from parameter accuracy
- recognize poor calibration-board coverage
- understand why low RMS alone does not prove good calibration
- save calibration output with units and model metadata

## 1. What calibration estimates

For one camera, calibration commonly estimates a shared intrinsic model:

$$
K =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

plus lens distortion:

$$
[k_1,k_2,p_1,p_2,k_3,\ldots]
$$

Each calibration image also has its own board pose:

$$
R_i,t_i
$$

So with N images:

~~~text
one shared K
one shared distortion model
N different extrinsic poses
~~~

## 2. Known target geometry

A checkerboard/chessboard target provides points whose relative geometry is known.

For a 9x6 inner-corner target with square spacing s:

$$
P_{r,c}
=
\begin{bmatrix}
(c-c_0)s \\
(r-r_0)s \\
0
\end{bmatrix}
$$

This lesson centers the synthetic board around its local origin.

The board is planar:

$$
Z=0
$$

in board coordinates.

## 3. Why multiple views?

A single image constrains the camera model only weakly.

Useful calibration data changes:

- board tilt
- board position
- board distance
- image location
- orientation

You want the target to exercise different parts of the camera model.

A dataset containing only nearly identical front-facing views can produce a numerically good-looking answer while leaving parameters poorly constrained.

## 4. Planar-calibration intuition

For a planar target, each image induces a homography between board coordinates and image coordinates.

Conceptually:

~~~text
known plane geometry
      +
many perspective views
      |
      v
constraints on K
      |
      v
initial intrinsics/extrinsics
      |
      v
nonlinear refinement
~~~

OpenCV handles the estimation/refinement internally.

This lesson focuses on understanding the data, outputs, and residual evidence around that optimization.

## 5. Synthetic ground truth

To make the lesson testable, the executable first creates a camera whose true parameters are known.

Default synthetic ground truth:

~~~text
fx = 910 px
fy = 895 px
cx = width/2 + 4
cy = height/2 - 3

k1 = -0.10
k2 =  0.020
p1 =  0.0012
p2 = -0.0008
k3 =  0
~~~

Then it creates many board poses and projects the known board points into the image.

Optional Gaussian corner noise is added afterward.

Because ground truth is known, we can evaluate:

- reprojection accuracy
- focal-length recovery
- principal-point recovery
- distortion recovery

That is much stronger than saying calibrateCamera returned something.

## 6. Observation noise

Real corner locations are not exact.

Sources include:

- image noise
- blur
- target printing accuracy
- corner-detector localization
- motion blur
- rolling shutter
- target non-flatness
- lens-model mismatch

The CLI option:

~~~text
--noise 0.15
~~~

adds deterministic Gaussian pixel noise to the synthetic corner observations.

The zero-noise path is useful for mathematical regression testing.

The noisy path is more realistic for calibration-quality discussion.

## 7. Calibration call

This lesson uses OpenCV calibrateCamera() with:

~~~text
CALIB_FIX_K3
~~~

The synthetic ground truth also uses:

~~~text
k3 = 0
~~~

Fixing k3 keeps the teaching problem well-conditioned and prevents the optimizer from using an unnecessary higher-order term to absorb small synthetic noise.

Later exercises can release k3 and study over-parameterization.

## 8. Reprojection residual

After calibration, take one observed board point.

Using estimated K, distortion, and per-view R_i,t_i, project the known 3-D board point back into the image.

Let p_obs be the measured corner and p_proj be the model prediction.

The residual vector is:

$$
r_i = p_{proj}-p_{obs}
$$

The residual magnitude is:

$$
e_i = \|r_i\|_2
$$

in pixels.

## 9. Per-view RMSE

For one calibration image with M corners:

$$
RMSE_{view}
=
\sqrt{
\frac{1}{M}
\sum_{j=1}^{M}
e_j^2
}
$$

Per-view error matters because one bad frame can be hidden inside a good global average.

## 10. Global reprojection RMSE

Across all calibration observations:

$$
RMSE_{global}
=
\sqrt{
\frac{
\sum_i e_i^2
}{
N_{points}
}
}
$$

The executable computes this independently after calibration.

It also reports maximum residual because averages alone can hide outliers.

## 11. Optimizer RMS vs independently recomputed RMSE

OpenCV returns an RMS value from calibration.

The lesson also recomputes all projected corners afterward.

These values should be consistent.

Why recompute?

Because production code should not blindly trust one opaque scalar returned by an optimizer. Independent evidence makes behavior easier to debug.

## 12. Low reprojection error does NOT guarantee correct calibration

A low residual can coexist with a weak calibration.

Examples:

- board views have poor pose diversity
- target covers only the image center
- all views are at nearly the same depth
- distortion model is too flexible
- board dimensions are wrong but compensated elsewhere
- calibration target is warped
- all observations share a systematic bias
- rolling shutter is not modeled
- focus/zoom changes after calibration

Calibration quality needs:

~~~text
residual quality
+
parameter plausibility
+
dataset geometry
+
repeatability
~~~

## 13. Coverage visualization

The executable writes:

~~~text
01_calibration_view_coverage.png
~~~

Every synthetic observation from every view is drawn.

A good real calibration dataset should cover:

- center
- edges
- corners
- different board tilts
- different distances

Do not collect many nearly identical pictures from the center of the image and expect strong distortion estimation.

## 14. Residual visualization

The executable also writes:

~~~text
02_reprojection_residuals_view0.png
~~~

Green dots are observed calibration corners.

Red vectors are reprojection residuals magnified 25x.

Residual-vector patterns can be more informative than one RMSE value.

For example, structured radial residuals may suggest model mismatch.

## 15. Calibration output file

The executable writes:

~~~text
calibration_report.yml
~~~

containing:

- image dimensions
- board dimensions
- square spacing
- number of views
- observation-noise configuration
- ground-truth K
- estimated K
- ground-truth distortion
- estimated distortion
- optimizer RMS
- independently recomputed global RMSE
- maximum residual
- per-view RMSE

In a real system, also record:

- camera serial number
- lens identity
- focus setting
- temperature range if relevant
- calibration date
- calibration software version
- target identity
- units
- coordinate-frame conventions

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_camera_calibration
~~~

## Run

~~~bash
./build/cv9x_camera_calibration \
  --output-dir build/camera_calibration \
  --width 1280 \
  --height 720 \
  --board-cols 9 \
  --board-rows 6 \
  --square-size 0.04 \
  --views 12 \
  --noise 0.15 \
  --seed 42
~~~

## Outputs

~~~text
01_calibration_view_coverage.png
02_reprojection_residuals_view0.png
calibration_report.yml
~~~

The console prints:

- ground-truth K
- estimated K
- ground-truth distortion
- estimated distortion
- optimizer RMS
- recomputed global RMSE
- maximum residual
- one RMSE per view

## Deterministic self-test

~~~bash
./build/cv9x_camera_calibration --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies two calibration regimes.

### Zero-noise case

It checks that:

- calibration completes
- optimizer RMS is near zero
- independently recomputed RMSE is near zero
- fx/fy recover close to ground truth
- cx/cy recover close to ground truth
- k1/k2/p1/p2 recover close to ground truth

### Noisy case

With 0.20-pixel corner noise it checks that:

- calibration still completes
- global reprojection error stays subpixel
- maximum residual stays bounded
- focal lengths remain near truth
- one RMSE is produced per view

The self-test does not require exact parameter equality under noise.

## Common failure cases

### Wrong square size

This changes the scale of translation estimates.

### Wrong corner count

A board described as 9x6 means 9x6 inner corners, not squares.

### Poor view diversity

This can make calibration parameters weakly constrained.

### Target occupies only the image center

Distortion near the edges becomes poorly observed.

### Target is not flat

The planar model is violated.

### Reusing stale calibration after camera changes

Changing lens, focus, sensor crop, resolution, or image pipeline can invalidate calibration.

### Judging quality from one RMS number

Inspect per-view residuals, parameter plausibility, image coverage, and repeatability.

## Engineering notes

Calibration should be treated as versioned sensor configuration, not as a random YAML file.

A production calibration contract should define:

~~~text
camera identity
image mode / resolution
intrinsic matrix
distortion model
coefficient ordering
coordinate-frame convention
units
valid operating conditions
calibration provenance
quality metrics
version
~~~

Consumers should reject incompatible calibration rather than silently applying it.

## Exercises

1. Sweep observation noise from 0 to 1 pixel and plot parameter error.
2. Use only front-facing views and compare parameter stability.
3. Restrict all corners to the image center and inspect distortion recovery.
4. Release k3 and compare overfitting behavior.
5. Remove one high-error view and recalibrate.
6. Implement chessboard detection on real images and feed those observations into the same residual-analysis pipeline.
7. Repeat calibration five times with different image subsets and measure parameter repeatability.

## Next lesson

Episode 18 / PR #20 covers stereo vision and epipolar geometry:

- two-camera geometry
- baseline
- essential and fundamental matrices
- epipolar constraint
- rectification
- disparity
- depth from disparity

See VIDEO.md.
