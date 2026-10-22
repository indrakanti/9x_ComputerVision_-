# Advanced Tracking 01 — SORT-Style Multi-Object Tracking

The original classical tracking lesson intentionally focused on one target:

~~~text
template correlation
+
Kalman prediction/correction
+
Tracking / Coasting / Lost lifecycle
~~~

This extension adds the missing multi-object layer:

~~~text
detections
    |
predict existing tracks
    |
build track-to-detection cost matrix
    |
IoU gate + Mahalanobis gate
    |
Hungarian assignment
    |
+-----------------------------+
| matched                     | unmatched
v                             v
Kalman correction         track miss / new birth
|                             |
v                             v
persistent ID            coast / delete / create
~~~

This is the core architecture behind SORT-style tracking.

## Learning objectives

By the end of this lesson you should be able to:

- distinguish detection from tracking
- explain why multi-object tracking requires data association
- build a track-to-detection cost matrix
- explain Hungarian assignment
- use IoU gating
- use Mahalanobis gating
- explain why gating should happen before accepting an assignment
- maintain persistent track IDs
- implement tentative/confirmed/coasting/deleted lifecycle
- handle track birth and death
- evaluate tracking with IoU, center error, MOTA, ID switches and IDF1
- connect a detector CSV to the tracker
- overlay resulting IDs on a video

## 1. Why single-object tracking is not enough

The original lesson knows exactly which target it is following.

Multi-object tracking has a harder problem.

At frame k-1:

~~~text
Track 1
Track 2
Track 3
~~~

At frame k a detector returns:

~~~text
Detection A
Detection B
Detection C
~~~

The tracker must answer:

> Which new detection belongs to which existing track?

That is the data-association problem.

## 2. SORT-style architecture

SORT popularized a very clean baseline:

~~~text
detector
   |
Kalman prediction
   |
Hungarian association
   |
Kalman correction
   |
track lifecycle
~~~

This lesson follows that architecture while making the important pieces explicit.

It does not copy a SORT implementation.

## 3. Detection interface

The tracker accepts a generic detection list.

CSV format:

~~~text
frame,x,y,w,h,score,class_id
~~~

Example:

~~~text
0,100,80,40,60,0.92,1
0,250,90,38,58,0.88,1
1,103,81,40,60,0.94,1
~~~

This deliberately separates:

~~~text
detection
from
tracking
~~~

PR #29 can therefore feed MOG2/blob detections into this tracker.

A later ML detector can use exactly the same interface.

## 4. Track state

Each track stores:

~~~text
persistent track ID
Kalman filter
estimated box size
age
successful-hit count
missed-frame count
lifecycle state
~~~

Kalman motion state:

$$
x=
\begin{bmatrix}
c_x\\
c_y\\
v_x\\
v_y
\end{bmatrix}
$$

Measurement:

$$
z=
\begin{bmatrix}
c_x\\
c_y
\end{bmatrix}
$$

Box width and height are carried from the most recent accepted detection.

## 5. Predict before association

Every existing track predicts where it expects to be in the next frame.

That produces:

~~~text
predicted Track 1 box
predicted Track 2 box
...
~~~

Association compares detections against these predictions rather than only against the previous measured positions.

This is especially important when objects move quickly or detections are temporarily missing.

## 6. Intersection over Union

For boxes A and B:

$$
IoU(A,B)
=
\frac{|A\cap B|}
{|A\cup B|}
$$

Properties:

~~~text
0 = no overlap
1 = identical boxes
~~~

The association cost used for an allowed pair is:

$$
cost=1-IoU
$$

So larger overlap means lower assignment cost.

## 7. IoU gating

Not every track should be allowed to match every detection.

If:

$$
IoU < IoU_{gate}
$$

the pair is forbidden.

This avoids assignments between geometrically implausible boxes.

However, IoU alone is not enough.

A fast-moving track can have modest overlap even when the measured center is statistically plausible.

That motivates a motion-uncertainty gate.

## 8. Mahalanobis gating

For innovation:

$$
y=z-H\hat{x}^{-}
$$

and innovation covariance:

$$
S=HP^{-}H^T+R
$$

the squared Mahalanobis distance is:

$$
d^2=y^TS^{-1}y
$$

This differs from ordinary Euclidean distance because it accounts for prediction uncertainty.

A track with high uncertainty can tolerate a larger absolute displacement than a very certain track.

The lesson accepts a track/detection pair only when:

$$
d^2 \le d^2_{gate}
$$

Default:

~~~text
9.21
~~~

This is a useful 2-D gating value for teaching, but production thresholds must be justified for the actual estimator and measurement model.

## 9. Two gates, two questions

IoU gate asks:

> Do the boxes overlap enough geometrically?

Mahalanobis gate asks:

> Is the measurement consistent with the track's predicted state uncertainty?

A pair must satisfy both.

## 10. Cost matrix

Suppose there are three tracks and four detections.

The tracker builds:

~~~text
             D0      D1      D2      D3
Track 1     0.1     INF     INF     0.7
Track 2     INF     0.2     0.5     INF
Track 3     INF     0.6     0.1     INF
~~~

Forbidden pairs get a very large cost.

The Hungarian algorithm then finds a globally minimum-cost one-to-one assignment.

## 11. Why greedy nearest-neighbor can fail

A greedy algorithm can make the locally best match first and leave a terrible match for another track.

Hungarian assignment minimizes the total assignment cost across the whole matrix.

That becomes important when tracks are close or cross.

## 12. Hungarian algorithm

The implementation is included directly in the C++ lesson.

It handles rectangular track/detection matrices by padding to a square matrix.

Complexity is approximately:

$$
O(n^3)
$$

for n assignments.

For modest object counts this is usually practical.

Large-scale systems may require more specialized association strategies.

## 13. Track birth

An unmatched detection creates a new tentative track.

The track receives a persistent integer ID.

Example:

~~~text
Detection unmatched
    |
new Track 7
state = tentative
hits = 1
~~~

A single noisy detection should not immediately become a trusted persistent object.

## 14. Tentative to confirmed

Default:

~~~text
min_hits = 2
~~~

The track becomes confirmed after enough successful associations.

A tentative track that immediately misses is deleted.

This suppresses many one-frame false detections.

## 15. Coasting

A confirmed track can temporarily survive without a detection.

During a miss:

~~~text
measurement = absent
Kalman prediction = available
state = coasting
missed += 1
~~~

The box remains a prediction, not a visual measurement.

## 16. Track death

Default:

~~~text
max_missed = 3
~~~

If:

$$
missed > maxMissed
$$

the track is deleted.

The ID is not reused.

This is important for downstream identity semantics.

## 17. Crossing objects

The deterministic sequence contains two objects moving toward and through each other.

Detection ordering is deliberately shuffled every frame.

The tracker cannot rely on:

~~~text
detection index == object identity
~~~

It must preserve identity through prediction and assignment.

## 18. Missed detections

The synthetic detector intentionally omits several objects for short intervals.

Confirmed tracks must:

1. coast using the motion model
2. remain alive within the miss budget
3. reassociate when the detection returns

## 19. False detections

The sequence also contains isolated false positives.

Because new tracks begin tentative and require multiple hits, one-frame false detections should not become persistent output IDs.

## 20. External detections

Run with a detector CSV:

~~~bash
./build/cv9x_multi_object_tracking \
  --detections detections.csv \
  --output-dir build/mot
~~~

Optional ground truth:

~~~bash
./build/cv9x_multi_object_tracking \
  --detections detections.csv \
  --truth truth.csv \
  --output-dir build/mot
~~~

Truth format:

~~~text
frame,id,x,y,w,h
~~~

Optional video overlay:

~~~bash
./build/cv9x_multi_object_tracking \
  --detections detections.csv \
  --truth truth.csv \
  --video input.mp4 \
  --output-dir build/mot
~~~

## 21. Tracking outputs

The tracker writes:

~~~text
mot_tracks.csv
tracking_summary.txt
multi_object_tracking.avi
~~~

Synthetic mode automatically creates the demonstration video.

Track CSV fields:

~~~text
frame
track_id
state
x
y
w
h
age
hits
missed
~~~

## 22. Per-match IoU

Mean matched IoU answers:

> When a predicted track was associated with ground truth, how well did the boxes overlap?

This measures localization quality, not identity persistence.

## 23. Center error

For matched box centers:

$$
e_c=
\sqrt{
(c_x-\hat c_x)^2+
(c_y-\hat c_y)^2
}
$$

Mean center error is useful for intuitive pixel-space accuracy.

## 24. MOTA

Multi-Object Tracking Accuracy is:

$$
MOTA=
1-
\frac{
FN+FP+IDSW
}{
GT
}
$$

where:

- FN = false negatives
- FP = false positives
- IDSW = identity switches
- GT = total ground-truth object instances

MOTA combines several failure types.

It does not directly measure identity precision the way IDF1 does.

## 25. Identity switch

If ground-truth object 2 was associated with Track 4 and later becomes associated with Track 9, that is an identity switch.

Crossing objects often expose this failure.

## 26. IDF1

IDF1 evaluates identity preservation across time:

$$
IDF1=
\frac{2IDTP}
{2IDTP+IDFP+IDFN}
$$

This lesson builds an identity-pair count matrix over the full sequence and uses Hungarian assignment again to obtain a global ground-truth-ID ↔ tracker-ID mapping.

That is more meaningful than only looking at per-frame overlap.

## 27. Evaluation matching vs tracking matching

These are separate operations.

Tracking association:

~~~text
predicted tracks
vs
detector outputs
~~~

Evaluation association:

~~~text
tracker outputs
vs
ground truth
~~~

Do not accidentally use ground truth to perform tracking association.

That would leak evaluation information into the algorithm.

## 28. Self-test

Run:

~~~bash
./build/cv9x_multi_object_tracking --self-test
~~~

The tests verify:

- identical boxes have IoU 1
- separated boxes have IoU 0
- Hungarian solver finds a known optimum
- a track stays tentative before minimum hits
- a repeated detection confirms the same persistent ID
- excessive misses delete the track
- synthetic crossing/missed-detection sequence has bounded mean IoU
- center error remains bounded
- MOTA passes a minimum threshold
- IDF1 passes a minimum threshold
- identity switches remain bounded

## 29. What this lesson intentionally does not do

It does not yet contain:

- MOG2 detection
- blob extraction
- MeanShift/CamShift
- EKF nonlinear motion
- NEES consistency
- appearance embeddings
- ReID
- MOSSE/KCF/CSRT
- learned detector integration

Those are the next advanced-tracking extensions.

## Engineering notes

A production track interface should usually distinguish:

~~~text
track ID
track state
measured vs predicted
box / state estimate
velocity
age
time since update
timestamp
measurement source
association score
covariance / uncertainty
class
confidence
model/config version
~~~

Persistent IDs are system interfaces, not merely labels drawn on a video.

## Exercises

1. Remove Mahalanobis gating and increase object speed.
2. Remove IoU gating and add nearby false detections.
3. Replace Hungarian assignment with greedy matching and compare ID switches.
4. Sweep min_hits.
5. Sweep max_missed.
6. Increase detector noise.
7. Increase missed-detection duration.
8. Add a fourth crossing object.
9. Add class-aware association.
10. Add an appearance cost term.

## Next advanced tracking lesson

**Background subtraction + blob detection + MeanShift/CamShift** will produce detections/tracking observations from actual image data and feed them into this multi-object framework.
