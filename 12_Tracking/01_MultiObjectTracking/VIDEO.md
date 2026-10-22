# Advanced Tracking A1 — Multi-Object Tracking with Hungarian Assignment

Suggested title:

**Multi-Object Tracking in C++ — SORT, Hungarian Assignment, Kalman Filters & IDF1**

Suggested thumbnail:

**DETECTIONS → IDs**

Target duration: 38–45 minutes.

## Chapters

~~~text
00:00 Single-object vs multi-object tracking
03:00 Detection-to-track association
07:00 IoU cost
11:00 Mahalanobis gating
15:00 Hungarian assignment
21:00 Track birth / confirmation / coasting / death
26:00 Crossing objects and identity switches
31:00 MOTA
35:00 IDF1
39:00 CSV detector interface
42:00 Next: MOG2 + CamShift
~~~

## Demo

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_multi_object_tracking

./build/cv9x_multi_object_tracking \
  --output-dir build/mot

./build/cv9x_multi_object_tracking --self-test
~~~

Show:

- shuffled detector order
- two crossing objects
- missed detections
- tentative false-positive tracks
- persistent IDs
- coasting
- MOTA/IDF1 summary

## Teaching point

A detector answers:

> What objects exist in this frame?

A tracker answers:

> Which current detection corresponds to which persistent object over time?

The second question is an assignment and state-estimation problem, not just a bounding-box drawing problem.
