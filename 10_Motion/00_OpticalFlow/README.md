# Module 15 — Optical Flow & Motion Estimation

Stereo correspondence matches points across space. Optical flow matches visual structure across time.

This is the first course module that treats images as an ordered temporal sequence instead of isolated frames.

~~~text
frame t
  |
  v
trackable corners
  |
  v
brightness constancy
  |
  v
spatial + temporal gradients
  |
  v
Lucas-Kanade local motion solve
  |
  v
image pyramids for larger motion
  |
  v
forward/backward validation
  |
  v
persistent feature trajectories
~~~

The executable supports a deterministic synthetic frame pair, two supplied image frames, or a video file with persistent feature IDs and bounded trajectory history.

## Learning objectives

By the end of this lesson you should be able to:

- explain optical flow as apparent image motion
- derive brightness constancy and the optical-flow constraint
- explain why one pixel cannot determine two motion components
- connect Lucas-Kanade to the same structure tensor used by corner detection
- explain the aperture problem
- solve the local 2x2 Lucas-Kanade normal equations
- explain why pyramids are needed for larger motion
- track features with pyramidal Lucas-Kanade
- perform forward/backward consistency validation
- maintain feature IDs and trajectories through a video stream
- understand track loss and re-detection

## 1. Brightness constancy

For a point at \((x,y)\) in frame \(t\), suppose the image displacement is \((u,v)\).

The classical assumption is:

$$
I(x,y,t)=I(x+u,y+v,t+\Delta t)
$$

Using a first-order Taylor approximation:

$$
I_xu+I_yv+I_t=0
$$

This is the optical-flow constraint equation.

It is approximate and can fail under lighting changes, specularities, motion blur, exposure changes, occlusion, and non-Lambertian surfaces.

## 2. Why one pixel is not enough

At one pixel:

~~~text
one equation
two unknowns: u and v
~~~

Lucas-Kanade assumes a small neighborhood shares approximately the same motion.

For many pixels:

$$
A
\begin{bmatrix}
u\\
v
\end{bmatrix}
=b
$$

The least-squares normal equations are:

$$
A^TA
\begin{bmatrix}
u\\
v
\end{bmatrix}
=A^Tb
$$

with:

$$
A^TA=
\begin{bmatrix}
\sum I_x^2 & \sum I_xI_y\\
\sum I_xI_y & \sum I_y^2
\end{bmatrix}
$$

This is the same local second-moment / structure-tensor idea used in corner detection.

That is why corners are good tracking features.

## 3. Aperture problem

A long straight edge has strong variation in one direction but little in the other.

One eigenvalue of the local structure tensor becomes small, making 2-D motion ambiguous.

The explicit Lucas-Kanade solver in this lesson rejects such singular/poorly conditioned patches.

## 4. Manual Lucas-Kanade

The function:

~~~text
solveLucasKanadeNormalEquations()
~~~

accumulates:

$$
G_{xx}=\sum I_x^2,\quad
G_{xy}=\sum I_xI_y,\quad
G_{yy}=\sum I_y^2
$$

and:

$$
b_x=-\sum I_xI_t,\quad
b_y=-\sum I_yI_t
$$

then solves the 2x2 system explicitly.

The lesson also visualizes a simple one-step, single-scale Lucas-Kanade estimate around detected corners.

That implementation is educational. The practical tracker is pyramidal.

## 5. Why pyramids matter

The Taylor expansion assumes motion is small.

If a feature moves many pixels, a single-scale local solve can fail.

Image pyramids reduce the apparent displacement at coarser levels:

~~~text
full resolution:       16 px motion
downsample by 2:        8 px
downsample by 4:        4 px
~~~

Pyramidal LK estimates at coarse scale and refines toward full resolution.

The practical tracker uses OpenCV calcOpticalFlowPyrLK() with configurable window size and pyramid level.

## 6. Feature selection

The practical pipeline starts with goodFeaturesToTrack().

This deliberately connects earlier lessons:

~~~text
corner detector
   |
structure tensor
   |
trackability
   |
Lucas-Kanade
~~~

## 7. Forward/backward validation

Forward tracking alone can drift or converge to incorrect texture.

The lesson tracks:

~~~text
frame 0 -> frame 1
~~~

then tracks the result backward:

~~~text
frame 1 -> frame 0
~~~

The forward/backward error is:

$$
e_{FB}=\|p_0-\hat p_0\|_2
$$

A track is retained only when:

$$
e_{FB}\le\tau_{FB}
$$

Default threshold:

~~~text
1.0 pixel
~~~

## 8. Deterministic synthetic motion

Without supplied frames, the executable generates a textured image and translates it by:

~~~text
dx = 8 px
dy = 5 px
~~~

by default.

It reports:

- detected features
- surviving forward/backward-valid tracks
- retention ratio
- median horizontal/vertical motion
- median forward/backward error

## 9. Pair mode

Run:

~~~bash
./build/cv9x_optical_flow \
  --frame1 frame_0001.png \
  --frame2 frame_0002.png \
  --output-dir build/flow
~~~

Outputs:

~~~text
01_detected_features.png
02_all_forward_tracks.png
03_fb_validated_tracks.png
04_manual_single_scale_lk.png
~~~

The manual single-scale view is intentionally less capable on larger motion than the pyramidal result.

## 10. Video mode

Run:

~~~bash
./build/cv9x_optical_flow \
  --video input.mp4 \
  --output-dir build/flow_video
~~~

Each persistent feature maintains:

~~~text
track ID
current position
bounded trajectory history
~~~

For every frame:

1. track current features forward
2. track them backward
3. reject inconsistent tracks
4. preserve IDs for surviving tracks
5. append trajectory history
6. remove lost tracks
7. detect replacement corners when the active count becomes low
8. assign new IDs
9. write the annotated frame

Output:

~~~text
tracking_overlay.avi
~~~

## 11. Track birth and death

Tracks are temporary.

They can be lost due to:

- occlusion
- image exit
- blur
- texture loss
- illumination change
- tracking failure

New features must be detected periodically.

This is the first course lesson with explicit temporal identity and state lifecycle.

## 12. Optical flow vs feature matching

Descriptor matching is useful for wider viewpoint/time gaps and non-local correspondence.

Optical flow is especially useful for temporally adjacent frames where appearance changes are limited.

A practical system can combine them:

~~~text
optical flow for frame-to-frame tracking
+
descriptor matching for reacquisition / relocalization
~~~

## 13. Optical flow vs stereo

Stereo is correspondence across cameras at approximately the same time.

Optical flow is correspondence across time, usually in the same camera.

Both depend on correspondence quality, validation, and geometry.

## 14. Units and timestamps

Optical flow is naturally:

~~~text
pixels per frame
~~~

With frame interval \(\Delta t\):

~~~text
pixels per second = pixels per frame / seconds per frame
~~~

Metric object velocity requires more information, such as camera calibration, depth, pose, and object geometry.

A production video pipeline should retain timestamps and dropped-frame information.

## 15. Deterministic self-test

Run:

~~~bash
./build/cv9x_optical_flow --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

### Exact normal-equation solve

Synthetic gradient/time samples are generated from known motion:

~~~text
u = 1.25
v = -0.75
~~~

The explicit 2x2 solver must recover that motion exactly.

### Aperture problem

A patch with gradients only in one direction must be rejected as singular.

### Pyramidal tracking

A deterministic image is translated by:

~~~text
dx = 7
dy = 4
~~~

The test requires:

- many detected corners
- many forward/backward-valid tracks
- median motion close to ground truth
- low median forward/backward error

The test does not require identical per-feature flow values across OpenCV builds.

## Common failure cases

### Tracking a straight edge

One motion component is ambiguous.

### Using no pyramid for large motion

The small-motion approximation breaks.

### Trusting only the forward status flag

Use forward/backward and application-specific checks.

### Never re-detecting

A long-running video tracker eventually runs out of active tracks.

### Re-detecting without preserving IDs

You lose temporal identity and trajectory history.

### Treating flow as meters/second

Raw flow is image displacement.

### Ignoring frame timing

Velocity depends on \(\Delta t\).

### Ignoring occlusion

A surface that is no longer visible cannot be tracked reliably.

### Unbounded trajectory storage

Bound track history to control memory.

## Engineering notes

A production feature-track message should define fields such as:

~~~text
track_id
frame_timestamp
pixel_x
pixel_y
age
status
forward_error
forward_backward_error
optional descriptor
optional depth
optional covariance / confidence
~~~

A production tracker should expose:

~~~text
input frame ID
tracks entering
tracks surviving
tracks rejected
new tracks
processing latency
dropped-frame count
configuration/version
~~~

This makes temporal failures diagnosable.

## Build

The top-level build requires the OpenCV video and videoio modules.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_optical_flow
~~~

## Synthetic demo

~~~bash
./build/cv9x_optical_flow \
  --output-dir build/optical_flow \
  --synthetic-dx 8 \
  --synthetic-dy 5 \
  --window 21 \
  --levels 3 \
  --fb-threshold 1.0
~~~

## Video demo

~~~bash
./build/cv9x_optical_flow \
  --video path/to/video.mp4 \
  --output-dir build/optical_flow_video \
  --max-frames 300 \
  --redetect-below 80 \
  --trajectory-length 30
~~~

## Exercises

1. Set pyramid levels to zero and increase synthetic motion until tracking degrades.
2. Sweep the forward/backward threshold and measure retention vs motion error.
3. Replace Shi-Tomasi features with FAST.
4. Fit a global affine transform with RANSAC over valid tracks.
5. Separate global camera motion from independently moving features.
6. Add timestamps and report pixels/second.
7. Add track age and reject very short-lived tracks.
8. Add descriptor-based reacquisition after track loss.
9. Attach stereo depth to tracked features and estimate approximate 3-D point motion.

## Next lesson

The natural roadmap now moves into classical recognition and tracking.

Module 16 covers contours, connected components, and shape descriptors before later object detection and tracking modules.
