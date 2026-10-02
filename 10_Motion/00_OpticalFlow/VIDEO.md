# Video: Optical Flow — Lucas-Kanade, Pyramids & Feature Tracking

YouTube: TBD

Suggested title:

**Optical Flow in C++ — Lucas-Kanade, Pyramids & Feature Tracking Explained**

Suggested thumbnail:

**HOW PIXELS MOVE**

Target duration: **32–38 minutes**

## Learning objectives

Viewers should understand:

- brightness constancy
- the optical-flow constraint
- why one pixel is underdetermined
- Lucas-Kanade least squares
- the aperture problem
- why corners track well
- why image pyramids recover larger motion
- forward/backward validation
- persistent feature IDs and trajectories in video

## Chapters

~~~text
00:00 Stereo was space — optical flow is time
02:00 Brightness constancy
05:00 Taylor expansion and flow equation
08:00 Why one pixel cannot solve 2-D motion
10:00 Lucas-Kanade normal equations
14:00 Aperture problem
17:00 Why corners are good trackers
19:00 Why single-scale LK fails on large motion
22:00 Pyramidal Lucas-Kanade
25:00 Forward/backward consistency
28:00 Synthetic translation demo
31:00 Video tracking with persistent IDs
35:00 Failure modes and engineering notes
37:00 Next: classical recognition
~~~

## Script outline

### 00:00 — Space vs time

Put the previous stereo lesson beside this one:

~~~text
stereo:
same time, different cameras

optical flow:
same camera, different times
~~~

Say:

> We have already learned correspondence across space. Now we begin correspondence across time.

### 02:00 — Brightness constancy

Show a corner moving a few pixels.

Write:

$$
I(x,y,t)=I(x+u,y+v,t+\Delta t)
$$

Discuss when this assumption is approximately valid and when it fails.

### 05:00 — Taylor expansion

Expand to first order:

$$
I_xu+I_yv+I_t=0
$$

Explain the three image derivatives.

### 08:00 — One equation, two unknowns

Write:

~~~text
known:
Ix Iy It

unknown:
u v
~~~

Explain why one pixel cannot determine 2-D motion.

### 10:00 — Lucas-Kanade

Assume one small patch shares a common motion.

Build the least-squares system.

Show:

$$
A^TA=
\begin{bmatrix}
\sum I_x^2 & \sum I_xI_y\\
\sum I_xI_y & \sum I_y^2
\end{bmatrix}
$$

Then point out:

> We have seen this matrix before.

Connect it to Harris/Shi-Tomasi.

### 14:00 — Aperture problem

Animate a long edge moving.

Show that motion along the edge is ambiguous.

Then run the singular-patch self-test.

### 17:00 — Why corners track

Compare:

- flat region
- edge
- corner

Use eigenvalue intuition from the corner lesson.

### 19:00 — Small-motion limitation

Translate a feature far enough that local linearization fails.

Show the manual one-step LK visualization.

### 22:00 — Pyramids

Draw a Gaussian pyramid.

Show how:

~~~text
16 px at level 0
8 px at level 1
4 px at level 2
~~~

Then explain coarse-to-fine refinement.

### 25:00 — Forward/backward check

Track:

~~~text
p0 -> p1 -> p0_hat
~~~

Write:

$$
e_{FB}=\|p_0-\hat p_0\|
$$

Explain why this catches many drift/failure cases.

### 28:00 — Synthetic demo

Run:

~~~bash
./build/cv9x_optical_flow \
  --output-dir build/optical_flow \
  --synthetic-dx 8 \
  --synthetic-dy 5
~~~

Show:

- detected features
- all forward tracks
- validated tracks
- manual single-scale LK

Read the median recovered motion from the console.

### 31:00 — Video mode

Run:

~~~bash
./build/cv9x_optical_flow \
  --video input.mp4 \
  --output-dir build/flow_video
~~~

Explain:

- persistent track ID
- track history
- track loss
- periodic re-detection
- new track IDs

Show tracking_overlay.avi.

### 35:00 — Engineering notes

Discuss:

- timestamps
- dropped frames
- track age
- bounded history
- occlusion
- illumination
- motion blur
- feature replenishment
- confidence / FB error

### 37:00 — Next

Transition from feature motion to higher-level classical recognition.

Preview contours, connected components, shape descriptors, and later object tracking.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_optical_flow

./build/cv9x_optical_flow \
  --output-dir build/optical_flow \
  --synthetic-dx 8 \
  --synthetic-dy 5 \
  --window 21 \
  --levels 3 \
  --fb-threshold 1.0

./build/cv9x_optical_flow --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**Why Corners Are Better Than Edges for Optical Flow**

30–45 seconds:

1. show a straight edge
2. explain the aperture problem
3. show the structure tensor
4. show a corner with strong gradients in two directions
5. close with: "good features to track are good because the motion solve is well-conditioned"

## YouTube description

This lesson starts the transition from static computer vision into temporal video processing.

We derive brightness constancy and the Lucas-Kanade equations, explain the aperture problem, connect feature tracking back to the structure tensor, use image pyramids for larger motion, validate tracks with a forward/backward check, and maintain persistent feature trajectories through video.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/10_Motion/00_OpticalFlow

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #OpticalFlow #LucasKanade #Tracking #Linux
