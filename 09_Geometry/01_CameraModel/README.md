# Module 12 — Camera Model: Intrinsics, Extrinsics, Distortion & 3-D → 2-D Projection

This lesson turns the projective geometry from the homography module into a physical camera model.

The full projection chain is:

~~~text
3-D world point
    |
    v
world -> camera transform
    |
    v
normalized pinhole coordinates
    |
    v
radial + tangential distortion
    |
    v
intrinsic matrix K
    |
    v
pixel coordinates
~~~

The implementation computes each stage explicitly and compares the final result against OpenCV \`projectPoints()\`.

## Learning objectives

By the end of this lesson you should be able to:

- distinguish world, camera, normalized-image, and pixel coordinates
- explain intrinsic vs extrinsic parameters
- write the camera intrinsic matrix \(K\)
- explain focal lengths \(f_x,f_y\) in pixels
- explain the principal point \((c_x,c_y)\)
- explain skew and why it is usually zero
- transform a world point into camera coordinates
- project a 3-D camera-frame point onto the normalized image plane
- apply radial distortion \(k_1,k_2,k_3\)
- apply tangential distortion \(p_1,p_2\)
- convert distorted normalized coordinates into pixels
- explain why points behind the camera are invalid
- understand the relationship between \(R,t\) and the physical camera center

## Coordinate convention

This lesson uses a **world-to-camera** transform:

$$
P_c = RP_w + t
$$

where:

- \(P_w\) = point in world coordinates
- \(P_c\) = point in camera coordinates
- \(R\) = world-to-camera rotation
- \(t\) = world-to-camera translation vector

This is an important convention.

The translation vector \(t\) is **not directly the camera position in world coordinates**.

The camera center in world coordinates is:

$$
C = -R^T t
$$

## 1. Pinhole projection

For camera-frame coordinates:

$$
P_c =
\begin{bmatrix}
X_c \\
Y_c \\
Z_c
\end{bmatrix}
$$

the normalized pinhole coordinates are:

$$
x = \frac{X_c}{Z_c}
$$

$$
y = \frac{Y_c}{Z_c}
$$

A valid visible point must satisfy:

$$
Z_c > 0
$$

under this lesson's convention.

## 2. Intrinsic matrix

The camera intrinsic matrix is:

$$
K =
\begin{bmatrix}
f_x & s & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
$$

where:

- \(f_x\) = focal length expressed in horizontal pixels
- \(f_y\) = focal length expressed in vertical pixels
- \(c_x,c_y\) = principal point
- \(s\) = skew

Most modern cameras use:

$$
s \approx 0
$$

## 3. Why focal length is in pixels

A physical focal length may be specified in millimeters.

But projection into a digital image requires conversion through pixel pitch / sensor dimensions.

That is why calibrated camera models usually expose \(f_x,f_y\) directly in pixel units.

## 4. Principal point

The principal point is where the optical axis intersects the image plane.

A common approximation is the image center:

$$
c_x \approx \frac{W}{2}
$$

$$
c_y \approx \frac{H}{2}
$$

but calibration estimates the actual values.

A point directly on the optical axis:

$$
(X_c,Y_c)=(0,0)
$$

projects to:

$$
(u,v)=(c_x,c_y)
$$

## 5. Extrinsics

Extrinsics describe the relationship between the world coordinate frame and the camera coordinate frame.

The lesson uses Euler angles to build:

$$
R = R_z(yaw)R_y(pitch)R_x(roll)
$$

and then:

$$
P_c = RP_w+t
$$

Euler angles are used here for teaching convenience.

Production systems often use:

- rotation matrices
- Rodrigues vectors
- quaternions
- Lie-group representations

depending on the application.

## 6. Radial distortion

Real lenses do not follow the ideal pinhole model perfectly.

Let:

$$
r^2=x^2+y^2
$$

The radial scale is:

$$
L(r)
=
1+k_1r^2+k_2r^4+k_3r^6
$$

Then:

$$
x_{radial}=xL(r)
$$

$$
y_{radial}=yL(r)
$$

Positive or negative coefficients can create barrel- or pincushion-like behavior depending on the lens/model.

## 7. Tangential distortion

Tangential distortion models decentering / lens misalignment effects.

The correction terms used in this lesson are:

$$
x_t =
2p_1xy
+
p_2(r^2+2x^2)
$$

$$
y_t =
p_1(r^2+2y^2)
+
2p_2xy
$$

So distorted normalized coordinates are:

$$
x_d=xL(r)+x_t
$$

$$
y_d=yL(r)+y_t
$$

## 8. Pixels

Finally:

$$
u=f_xx_d+s y_d+c_x
$$

$$
v=f_yy_d+c_y
$$

For the common zero-skew case:

$$
u=f_xx_d+c_x
$$

## 9. Complete projection equation

Ignoring lens distortion for a moment, the compact form is:

$$
\lambda
\begin{bmatrix}
u\\
v\\
1
\end{bmatrix}
=
K
\begin{bmatrix}
R&t
\end{bmatrix}
\begin{bmatrix}
X_w\\
Y_w\\
Z_w\\
1
\end{bmatrix}
$$

Lens distortion acts between normalized pinhole projection and application of \(K\).

## Implementation

The canonical implementation is:

~~~text
09_Geometry/01_CameraModel/camera_model.cpp
~~~

Core functions:

- \`rotationMatrix()\`
- \`intrinsicMatrix()\`
- \`worldToCamera()\`
- \`distortNormalized()\`
- \`projectPointManual()\`
- \`projectOpenCvReference()\`

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_camera_model
~~~

## Run

Example:

~~~bash
./build/cv9x_camera_model \
  --output-dir build/camera_model \
  --width 1280 \
  --height 720 \
  --fx 900 \
  --fy 895 \
  --cx 640 \
  --cy 360 \
  --roll 3 \
  --pitch -8 \
  --yaw 12 \
  --tx 0.10 \
  --ty -0.05 \
  --tz 0.40 \
  --k1 -0.12 \
  --k2 0.025 \
  --p1 0.001 \
  --p2 -0.001 \
  --k3 -0.003
~~~

## Outputs

~~~text
01_pinhole_projection.png
02_distorted_projection.png
~~~

The executable also prints:

- \(K\)
- \(R\)
- \(t\)
- distortion coefficients
- manual-vs-OpenCV maximum projection error

## Deterministic self-test

~~~bash
./build/cv9x_camera_model --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- optical-axis point projects to the principal point
- a simple pinhole projection matches hand calculation
- translation affects camera-frame projection as expected
- positive radial \(k_1\) moves an off-axis normalized point outward
- tangential distortion matches its explicit formula
- points behind the camera are rejected
- manual projection matches OpenCV \`projectPoints()\`
- the constructed rotation matrix is orthonormal
- rotation determinant is +1

## Common failure cases

### Confusing camera position with \(t\)

For:

$$
P_c=RP_w+t
$$

the camera center is:

$$
C=-R^Tt
$$

### Mixing transform directions

A world-to-camera transform is not interchangeable with a camera-to-world transform.

### Forgetting the division by \(Z_c\)

Perspective projection requires:

$$
x=X_c/Z_c,\quad y=Y_c/Z_c
$$

### Applying distortion after pixel scaling

The standard radial/tangential model operates in normalized image coordinates before applying \(K\).

### Using points behind the camera

A mathematically computed pixel does not mean the physical point is visible.

### Treating image center as guaranteed principal point

It is only an approximation until calibrated.

## Engineering notes

A production camera-model interface should explicitly define:

~~~text
coordinate-frame convention
transform direction
length units
pixel convention
intrinsic matrix
distortion model
distortion coefficient ordering
valid depth range
image resolution
calibration version
timestamp / sensor identity
numeric precision
~~~

These contracts become critical in stereo, pose estimation, visual odometry, sensor fusion, and production perception systems.

## Exercises

1. Compute camera center \(C=-R^Tt\) and print it.
2. Add camera-to-world input and convert it into world-to-camera form.
3. Visualize distortion vectors across the image plane.
4. Compare barrel vs pincushion coefficient sets.
5. Add a 3-D cube and draw its projected edges.
6. Add support for a fisheye model and compare with the pinhole model.
7. Sweep focal length and visualize field-of-view change.

## Next lesson

Episode 17 covers **camera calibration and reprojection error**:

- calibration target geometry
- multiple views
- estimating \(K\)
- estimating distortion
- per-image extrinsics
- reprojection residuals
- calibration quality and failure modes

See [VIDEO.md](VIDEO.md).
