# Advanced Tracking A2 — MOG2, Blob Detection, MeanShift & CamShift

Suggested title:

**Background Subtraction & CamShift in C++ — Detect Then Track with OpenCV**

Suggested thumbnail:

**PIXELS → BLOBS → TRACKS**

Target duration: 38–45 minutes.

## Chapters

~~~text
00:00 Detection vs tracking
03:00 Static-camera background subtraction
07:00 MOG2 mixture-model intuition
12:00 Foreground masks and shadows
16:00 Morphology
19:00 Connected components / blobs
23:00 Export detections into SORT
27:00 HSV histogram back-projection
31:00 MeanShift
35:00 CamShift
39:00 Scale-change comparison
42:00 Next: EKF + NEES
~~~

## Demo 1 — MOG2 to SORT

~~~bash
./build/cv9x_background_camshift \
  --output-dir build/mog2

./build/cv9x_multi_object_tracking \
  --detections build/mog2/mog2_detections.csv \
  --output-dir build/mot_from_mog2
~~~

Show that:

~~~text
MOG2 = object proposal / detection
SORT A1 = identity association + lifecycle
~~~

## Demo 2 — MeanShift vs CamShift

Open:

~~~text
05_hue_backprojection.png
meanshift_camshift_overlay.avi
appearance_tracking_metrics.csv
~~~

Explain why MeanShift keeps a fixed window and why CamShift expands as the target grows.

## Short idea

**Background Subtraction Is Detection, Not Tracking**

Show:

1. MOG2 mask
2. connected blobs
3. detection boxes without IDs
4. feed boxes into A1
5. persistent IDs appear

Close with:

> Detection tells you what exists now. Tracking tells you which object is which over time.
