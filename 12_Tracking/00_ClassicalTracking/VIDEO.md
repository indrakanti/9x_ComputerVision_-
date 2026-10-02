# Video: Classical Object Tracking — Correlation + Kalman Filter in C++

YouTube: TBD

Suggested title:

**Object Tracking in C++ — Template Correlation, Kalman Filter & Track Lifecycle**

Suggested thumbnail:

**MEASURE → PREDICT → TRACK**

Target duration: **32–38 minutes**

## Learning objectives

Viewers should understand:

- normalized cross-correlation
- local template search
- prediction vs measurement
- constant-velocity Kalman state
- Kalman predict/correct
- temporary coasting through occlusion
- Lost-state behavior
- template drift
- tracker evidence and lifecycle

## Chapters

~~~text
00:00 Detection vs tracking
02:00 Initialize an object ROI
05:00 Normalized cross-correlation
09:00 Local correlation search
12:00 Why prediction reduces search cost
14:00 Kalman state: x, y, vx, vy
18:00 Predict vs correct
22:00 Tracking / Coasting / Lost
26:00 Short occlusion demo
29:00 Long outage -> Lost
31:00 Template update and drift
34:00 Real video mode
36:00 Classical CV section complete
~~~

## Script outline

### 00:00 — Detector vs tracker

Show a single image, then a sequence.

Say:

> A detector starts from the image. A tracker starts from the image plus memory.

### 02:00 — Initial ROI

Draw a target box on the first frame and extract the template.

### 05:00 — NCC

Write:

$$
NCC=
\frac{
\sum(T-\bar T)(I-\bar I)
}{
\sqrt{
\sum(T-\bar T)^2
\sum(I-\bar I)^2
}
}
$$

Explain why subtracting means and normalizing energy helps brightness/contrast variation.

### 09:00 — Local search

Slide the template around the predicted location.

Find the highest score.

Then apply the acceptance threshold.

### 12:00 — Prediction reduces compute

Compare:

~~~text
full-frame search
vs
local search around prediction
~~~

### 14:00 — Kalman state

Write:

$$
[x,y,v_x,v_y]^T
$$

Show the constant-velocity transition matrix.

### 18:00 — Predict vs correct

Every new frame begins with prediction.

If a validated correlation measurement exists, correct the state.

If not, keep prediction only.

### 22:00 — Lifecycle

Draw:

~~~text
Initializing
    |
Tracking
    |
measurement missing
    v
Coasting
    |
too many misses
    v
Lost
~~~

Explain that Coasting is weaker evidence than Tracking.

### 26:00 — Short occlusion

Run the deterministic sequence.

The target disappears briefly.

The Kalman prediction carries the state.

When it reappears, correlation reacquires it.

### 29:00 — Long outage

Show the long-occlusion self-test.

Emphasize that a tracker must eventually admit that it no longer knows the target location.

### 31:00 — Template drift

Enable a small template update.

Explain both gradual adaptation and the risk of learning the wrong patch.

### 34:00 — Real video mode

Run:

~~~bash
./build/cv9x_classical_tracking \
  --video input.mp4 \
  --roi 180,90,48,72 \
  --output-dir build/tracking
~~~

Show the overlay, trajectory image, and CSV evidence.

### 36:00 — Classical section complete

Recap:

~~~text
pixels
-> filters
-> gradients
-> features
-> matching
-> geometry
-> cameras
-> stereo
-> motion
-> recognition
-> classification
-> tracking
~~~

Then transition to the separate ML Computer Vision repository for learned/deep vision.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_classical_tracking

./build/cv9x_classical_tracking \
  --output-dir build/classical_tracking

./build/cv9x_classical_tracking --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**A Predicted Bounding Box Is NOT a Measurement**

30–45 seconds:

1. target visible -> measurement-corrected track
2. target occluded -> prediction-only box
3. same coordinate type, different evidence
4. close with: tracking state must travel with the box

## YouTube description

This lesson completes the classical recognition and tracking section.

We implement normalized cross-correlation for template matching, use a constant-velocity Kalman filter to predict object motion, distinguish measurement correction from prediction-only coasting, and build explicit Initializing, Tracking, Coasting, and Lost states.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/12_Tracking/00_ClassicalTracking

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #ObjectTracking #KalmanFilter #TemplateMatching #Linux
