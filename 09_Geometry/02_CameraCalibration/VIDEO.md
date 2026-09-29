# Video: Camera Calibration in C++ — Intrinsics, Distortion & Reprojection Error

YouTube: TBD

Suggested title:

**Camera Calibration in C++ — Intrinsics, Distortion & Reprojection Error**

Suggested thumbnail:

**CAN YOU TRUST THIS CALIBRATION?**

Target duration: **30–36 minutes**

## Learning objectives

Viewers should understand:

- what camera calibration estimates
- why calibration needs multiple board poses
- how a planar target constrains a camera model
- what reprojection error means
- why per-view error matters
- why low RMS alone does not prove calibration quality

## Chapters

~~~text
00:00 We know the camera model — but how do we estimate it?
02:00 What calibration solves
05:00 Known board geometry
08:00 Why one image is not enough
11:00 Multiple perspective views
14:00 calibrateCamera inputs and outputs
17:00 Reprojection residual
20:00 Per-view vs global RMSE
23:00 Synthetic ground truth
26:00 Add corner noise
28:00 Coverage and residual visualization
31:00 Why low RMS can still lie
34:00 Next: stereo geometry
~~~

## Script outline

### 00:00 — Connect to camera-model lesson

Show:

~~~text
K
distortion
R,t
~~~

Then ask:

> Last time we assumed these were known. How do we actually obtain them?

### 02:00 — Unknowns

Separate:

~~~text
shared across all views:
K + distortion

different for every view:
R_i + t_i
~~~

### 05:00 — Calibration board

Draw a 9x6 inner-corner target.

Explain that its 3-D geometry is known.

Clarify:

> "9x6" means inner corners, not squares.

### 08:00 — Why one view fails

Show one perfectly front-facing board.

Then show several tilted views.

Explain that parameter observability improves when the target changes:

- tilt
- location
- distance
- orientation

### 11:00 — Planar calibration intuition

Show:

~~~text
board plane
 -> image homography
 -> constraints on camera intrinsics
 -> nonlinear refinement
~~~

Do not bury the viewer in the full derivation yet; connect it to the homography lessons they already completed.

### 14:00 — API contract

Walk through the inputs and outputs of OpenCV calibrateCamera():

~~~text
objectPoints
imagePoints
imageSize
K
distortion
rvecs
tvecs
~~~

Explain the shape of every collection.

### 17:00 — Residual

Draw:

- green observed corner
- projected model corner
- vector between them

Write:

$$
e_i = ||p_{proj}-p_{obs}||_2
$$

### 20:00 — RMSE

Show:

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum e_i^2
}
$$

Then distinguish:

~~~text
optimizer RMS
global recomputed RMSE
per-view RMSE
maximum residual
~~~

### 23:00 — Synthetic truth

Reveal why the lesson uses synthetic observations first.

We know:

- exact K
- exact distortion
- exact board geometry
- exact poses

So recovery quality can be tested.

### 26:00 — Add corner noise

Run once with:

~~~text
--noise 0
~~~

Then with:

~~~text
--noise 0.15
~~~

Compare parameter and residual behavior.

### 28:00 — Visual diagnostics

Show:

~~~text
01_calibration_view_coverage.png
02_reprojection_residuals_view0.png
~~~

Explain why target coverage matters.

### 31:00 — A low RMS can still lie

Give failure examples:

- same pose repeated many times
- only image center covered
- warped target
- wrong distortion model
- wrong board dimensions
- calibration invalidated after focus/lens change

### 34:00 — Next

Place two calibrated cameras side by side.

Preview:

> Next, calibration becomes stereo geometry: epipolar lines, disparity, and depth.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_camera_calibration

./build/cv9x_camera_calibration \
  --output-dir build/camera_calibration \
  --board-cols 9 \
  --board-rows 6 \
  --square-size 0.04 \
  --views 12 \
  --noise 0.15 \
  --seed 42

./build/cv9x_camera_calibration --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**Low Reprojection Error Does NOT Mean Good Calibration**

30–45 seconds:

1. show a low RMS
2. show board observations clustered at image center
3. say "the model fits what you showed it"
4. show edge distortion unobserved
5. close with: "quality starts with calibration geometry, not just the final RMS"

## YouTube description

This lesson explains camera calibration beyond a one-line OpenCV API call.

We generate known calibration-board observations, estimate camera intrinsics and lens distortion, recompute every reprojection residual, visualize calibration coverage, and explain why a low RMS alone does not prove that a calibration is trustworthy.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/09_Geometry/02_CameraCalibration

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #CameraCalibration #ReprojectionError #Linux
