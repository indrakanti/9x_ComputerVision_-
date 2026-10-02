# Module 19 — Classical Tracking: Correlation, Kalman Prediction & Track Lifecycle

This module completes the original **Classical Recognition and Tracking** section.

The previous optical-flow lesson tracked many local feature points. This lesson tracks one object-level region over time and makes the tracker lifecycle explicit.

~~~text
initial object ROI
      |
      v
template
      |
      v
Kalman prediction
      |
      v
local correlation search
      |
      +-------------------------+
      |                         |
      v                         v
good visual match            no valid match
      |                         |
      v                         v
measurement correction      prediction only
      |                         |
      v                         v
   Tracking                  Coasting
      |                         |
      +-----------+-------------+
                  |
          too many misses
                  |
                  v
                 Lost
~~~

The implementation uses:

- manual normalized cross-correlation
- local template search
- constant-velocity Kalman filtering
- explicit tracker states
- optional template adaptation
- deterministic synthetic occlusion tests
- real video input with an initial ROI

## Learning objectives

By the end of this lesson you should be able to:

- explain template matching as object-level appearance tracking
- derive normalized cross-correlation
- distinguish detection measurement from state prediction
- explain why local search is cheaper than full-frame search
- explain constant-velocity Kalman state
- distinguish predict and correct steps
- understand process noise vs measurement noise
- define Tracking, Coasting, and Lost states
- handle temporary occlusion
- understand template-update drift
- record tracker evidence over time
- explain when a classical tracker should be re-detected or terminated

## 1. Tracking is stateful

An object detector answers:

> Is the object visible in this frame, and where?

A tracker answers:

> Given what I believed in previous frames, where should the object be now, and does the current image support that belief?

Tracking therefore has memory.

A useful tracker separates:

~~~text
state prediction
measurement
measurement validation
state correction
lifecycle
~~~

## 2. Initial ROI

For the first frame we assume the target location is known:

~~~text
x, y, width, height
~~~

The image pixels inside that ROI form the initial template.

For real video mode, the ROI is supplied with:

~~~text
--roi x,y,w,h
~~~

In a production system, this initial box would often come from a detector, another perception module, or a prior track handoff.

## 3. Normalized cross-correlation

A simple dot product between patches changes strongly with brightness and contrast.

This lesson implements:

$$
NCC=
\frac{
\sum_i
(T_i-\bar T)
(I_i-\bar I)
}{
\sqrt{
\sum_i(T_i-\bar T)^2
\sum_i(I_i-\bar I)^2
}
}
$$

For identical non-constant patches:

$$
NCC=1
$$

Values closer to 1 indicate stronger similarity.

## 4. What normalization helps

Subtracting patch means reduces sensitivity to brightness offset.

Dividing by patch energy reduces sensitivity to overall contrast scale.

NCC is still sensitive to:

- rotation
- scale change
- large deformation
- severe viewpoint change
- occlusion

## 5. Local search

Searching every possible location every frame is expensive.

If motion is limited, search can be centered around the predicted location:

~~~text
predicted center +/- search radius
~~~

Default:

~~~text
28 pixels
~~~

A better prediction permits a smaller search region and therefore lower compute.

## 6. Measurement threshold

The highest-correlation candidate is not automatically valid.

The tracker accepts a visual measurement only when:

$$
NCC \ge \tau
$$

Default:

~~~text
0.72
~~~

If the threshold is not met, the frame provides no trusted image measurement.

A predicted position can still exist, but it is not the same evidence as a visual observation.

## 7. Kalman state

The constant-velocity state is:

$$
x_k=
\begin{bmatrix}
p_x\\
p_y\\
v_x\\
v_y
\end{bmatrix}
$$

where position is measured in pixels and velocity in pixels per frame.

The measurement is:

$$
z_k=
\begin{bmatrix}
p_x\\
p_y
\end{bmatrix}
$$

## 8. State transition

For one-frame time step:

$$
F=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}
$$

So:

$$
p_{x,k+1}=p_{x,k}+v_{x,k}
$$

$$
p_{y,k+1}=p_{y,k}+v_{y,k}
$$

## 9. Predict step

Before examining the next frame, the filter predicts the next state.

The predicted position defines the center of the correlation search region.

## 10. Correct step

When correlation produces a trusted measurement, the Kalman filter combines:

~~~text
prediction
+
measurement
+
uncertainty
~~~

to update the state.

## 11. Process noise

Process noise represents uncertainty in the motion model.

Real targets can accelerate, turn, stop, or change direction.

Too little process noise makes the model overconfident.

Too much makes prediction less stable.

## 12. Measurement noise

Measurement noise represents uncertainty in the image localization.

Sources include:

- correlation ambiguity
- blur
- partial occlusion
- deformation
- pixel quantization
- template mismatch

## 13. Lifecycle states

The lesson records four states.

### Initializing

The first ROI creates the template and initial state.

### Tracking

A valid visual measurement was accepted and the filter was corrected.

### Coasting

No valid visual measurement was found, but the miss budget has not been exceeded.

Prediction is published without a measurement correction.

### Lost

Too many consecutive measurements are missing.

The object identity/location should no longer be considered reliable.

## 14. Why Coasting matters

A short occlusion should not necessarily delete a track immediately.

For example, a target passing behind a narrow obstacle may disappear briefly.

During Coasting:

~~~text
visual measurement = unavailable
prediction         = available
confidence         = weaker
~~~

Downstream systems should know that distinction.

## 15. Lost threshold

Default:

~~~text
--max-missed 5
~~~

Each rejected/missing correlation measurement increments the miss count.

When misses exceed the configured limit, the state becomes Lost.

## 16. Short occlusion and reacquisition

The deterministic synthetic sequence hides the target briefly.

During the occlusion:

~~~text
correlation fails validation
Kalman prediction continues
state = Coasting
~~~

When the target reappears inside the search region:

~~~text
correlation passes threshold
measurement is accepted
Kalman correction runs
state returns to Tracking
~~~

The self-test requires this behavior.

## 17. Long outage

A second synthetic sequence hides the target longer than the miss budget.

The self-test requires the tracker to reach Lost.

This prevents a dangerous failure mode where plausible-looking predicted boxes continue indefinitely after visual evidence is gone.

## 18. Template update

By default the template is fixed:

~~~text
--template-alpha 0
~~~

Optional adaptation uses:

$$
T_{new}
=
(1-\alpha)T_{old}
+
\alpha I_{measurement}
$$

This can help gradual appearance change but introduces template drift risk.

If a wrong patch is blended into the template, the tracker can slowly learn background instead of the target.

## 19. Synthetic sequence

Without video input, the executable generates a deterministic sequence with:

- textured background
- one textured target
- constant 3 px/frame horizontal motion
- constant 2 px/frame vertical motion
- a short occlusion
- known ground-truth box each frame

## 20. Outputs

~~~text
01_initial_template.png
02_final_template.png
03_trajectory.png
tracking_metrics.csv
tracking_overlay.avi
~~~

The overlay shows:

- green = Tracking / initialized
- yellow = Coasting
- red = Lost
- orange = synthetic ground truth
- magenta point = Kalman predicted center

The CSV records frame state, estimated box, predicted/estimated centers, NCC score, miss count, whether a measurement was used, and synthetic truth when available.

## 21. Deterministic self-test

Run:

~~~bash
./build/cv9x_classical_tracking --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- identical-patch manual NCC equals 1
- manual NCC search finds a known translated template
- manual NCC matches OpenCV normalized correlation on an identical patch
- constant-velocity Kalman prediction advances by configured velocity
- one tracking record is emitted per frame
- visible-frame mean center error stays bounded
- short occlusion uses Coasting rather than Lost
- tracking reacquires after short occlusion
- a long measurement outage reaches Lost

## 22. Real video mode

Example:

~~~bash
./build/cv9x_classical_tracking \
  --video input.mp4 \
  --roi 180,90,48,72 \
  --output-dir build/classical_tracking
~~~

The ROI must lie inside the first frame.

## 23. Correlation tracking vs optical flow

Optical flow tracks local visual points.

Template correlation tracks one object-level patch.

They can be combined. For example, optical-flow points inside an object box can estimate object translation or deformation.

## 24. Tracking vs detection

Detection can recognize an object without prior location.

Tracking exploits temporal continuity.

A robust system often uses:

~~~text
detector -> initialize
tracker  -> frame-to-frame updates
detector -> periodic correction / reacquisition
~~~

This interaction is foundational for modern multi-object tracking.

## 25. Limitations

This simple tracker assumes:

- approximately constant target size
- limited rotation
- local inter-frame motion
- enough appearance texture
- limited deformation

It does not solve:

- scale adaptation
- large rotations
- long-term re-identification
- multi-object association
- crowded-scene identity management

Those naturally belong in later learned/deep tracking material.

## Engineering notes

A production track output should carry evidence, not only coordinates.

Example:

~~~text
track_id
timestamp
state
bbox
predicted_bbox
measurement_source
measurement_score
missed_count
age
velocity
covariance
last_measurement_time
model/config version
~~~

A downstream consumer may treat Tracking very differently from Coasting even if both contain numeric boxes.

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_classical_tracking
~~~

## Synthetic demo

~~~bash
./build/cv9x_classical_tracking \
  --output-dir build/classical_tracking \
  --search-radius 28 \
  --max-missed 5 \
  --ncc-threshold 0.72
~~~

## Video demo

~~~bash
./build/cv9x_classical_tracking \
  --video input.mp4 \
  --roi 180,90,48,72 \
  --output-dir build/classical_tracking_video \
  --max-frames 300 \
  --search-radius 36 \
  --ncc-threshold 0.65
~~~

## Exercises

1. Change process noise and inspect prediction during occlusion.
2. Change measurement noise and observe correction smoothness.
3. Sweep NCC threshold and measure false matches vs misses.
4. Enable template adaptation and induce drift.
5. Add scale search.
6. Add full-frame reacquisition after Lost.
7. Use optical-flow points inside the ROI.
8. Add two objects and persistent track IDs.
9. Add detector measurements as periodic corrections.
10. Use real timestamps instead of fixed one-frame time steps.

## Course boundary

With this module, the original classical Computer Vision flow through image processing, features, matching, geometry, cameras, stereo, motion, recognition, classification, and tracking is complete.

The next subject area is learned/deep computer vision, which can cleanly move into the separate ML Computer Vision repository.
