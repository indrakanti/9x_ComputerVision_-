# Video: Camera Intrinsics, Extrinsics & Distortion from First Principles

YouTube: TBD

Suggested title:

**Camera Intrinsics, Extrinsics & Distortion — 3-D to Pixels from First Principles**

Suggested thumbnail:

**3-D POINT → PIXEL**

Target duration: **28–34 minutes**

## Learning objectives

Viewers should understand:

- world vs camera coordinates
- \(R,t\) extrinsics
- pinhole projection
- \(K\) intrinsics
- principal point and focal length
- radial/tangential distortion
- full 3-D → 2-D projection chain

## Chapters

~~~text
00:00 What a camera model must answer
02:00 World vs camera coordinates
05:00 Extrinsics: R and t
08:00 Pinhole projection X/Z, Y/Z
11:00 Intrinsic matrix K
15:00 Principal point and focal length in pixels
18:00 Radial distortion
21:00 Tangential distortion
24:00 Full 3-D to pixel pipeline
27:00 C++ implementation
30:00 Deterministic tests
32:00 Next: camera calibration
~~~

## Script outline

### 00:00 — The question

Show a 3-D point floating in front of a camera.

Ask:

> Which pixel sees this point?

Then reveal the full chain:

~~~text
world
 -> camera
 -> normalized image
 -> distortion
 -> pixels
~~~

### 02:00 — Coordinate frames

Draw:

- world frame
- camera frame
- image plane

Explain why the same 3-D point has different coordinates in different frames.

### 05:00 — Extrinsics

Write:

$$
P_c=RP_w+t
$$

Then immediately clarify:

> \(t\) is not directly the camera center.

Show:

$$
C=-R^Tt
$$

### 08:00 — Pinhole projection

Write:

$$
x=X_c/Z_c
$$

$$
y=Y_c/Z_c
$$

Move the same point farther away and show the normalized coordinates move toward the center.

### 11:00 — Intrinsics

Write:

$$
K=
\begin{bmatrix}
f_x&s&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}
$$

Explain every parameter.

### 15:00 — Focal length in pixels

Connect physical optics to digital sampling.

Show why a camera model used by software normally reports focal lengths in pixels.

### 18:00 — Radial distortion

Draw ideal grid vs barrel/pincushion-like grid.

Write:

$$
1+k_1r^2+k_2r^4+k_3r^6
$$

### 21:00 — Tangential distortion

Explain decentering / misalignment.

Show the \(p_1,p_2\) terms without overcomplicating the physical optics.

### 24:00 — Complete projection

Walk through one point numerically:

~~~text
Pw
 -> R Pw + t
 -> divide by Z
 -> distort
 -> multiply by fx/fy
 -> add principal point
~~~

### 27:00 — Code

Walk through:

- \`rotationMatrix()\`
- \`worldToCamera()\`
- \`distortNormalized()\`
- \`projectPointManual()\`

Then compare with \`cv::projectPoints()\`.

### 30:00 — Tests

Run:

~~~bash
./build/cv9x_camera_model --self-test
~~~

Show:

- principal-point test
- hand-calculated projection
- radial/tangential checks
- OpenCV equivalence
- invalid point behind camera

### 32:00 — Next

Preview calibration:

> Today we assumed \(K\), distortion, \(R\), and \(t\) were known. Next we estimate them from images.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_camera_model

./build/cv9x_camera_model \
  --output-dir build/camera_model \
  --fx 900 --fy 895 --cx 640 --cy 360 \
  --roll 3 --pitch -8 --yaw 12 \
  --tx 0.10 --ty -0.05 --tz 0.40 \
  --k1 -0.12 --k2 0.025 \
  --p1 0.001 --p2 -0.001 --k3 -0.003

./build/cv9x_camera_model --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**Why Camera Translation t Is NOT the Camera Position**

30–45 seconds:

1. write \(P_c=RP_w+t\)
2. ask where the camera center is
3. set \(P_c=0\)
4. solve:
   \(C=-R^Tt\)
5. close with: "Always define transform direction."

## YouTube description

This lesson derives the complete pinhole camera model from first principles.

We transform 3-D world points into the camera frame, perform perspective division, apply radial and tangential lens distortion, apply the intrinsic matrix, and compare the manual C++ projection against OpenCV projectPoints.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/09_Geometry/01_CameraModel

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #CameraCalibration #ProjectiveGeometry #Linux
