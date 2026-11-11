# Advanced Tracking A2 — Background Subtraction, Blobs, MeanShift & CamShift

This lesson adds the classical image-level front end that was missing from the tracking path.

Two complementary pipelines are taught:

~~~text
STATIC-CAMERA DETECT → TRACK

video
  |
MOG2 background model
  |
foreground mask
  |
morphology
  |
connected components / blobs
  |
mog2_detections.csv
  |
A1 SORT-style multi-object tracker
  |
persistent IDs
~~~

and:

~~~text
APPEARANCE TRACKING

initial ROI
  |
HSV hue histogram
  |
histogram back-projection
  |
MeanShift / CamShift
  |
tracked window
~~~

## Learning objectives

You should be able to:

- explain background subtraction for a static camera
- understand MOG2 history, variance threshold, learning rate and shadows
- clean a foreground mask with morphology
- convert foreground regions into blob detections
- export detections into the A1 tracker
- build an HSV hue histogram
- compute histogram back-projection
- explain MeanShift mode seeking
- explain CamShift scale/orientation adaptation
- compare fixed-window and adaptive-window tracking

## 1. Detection and tracking are separate

MOG2 answers:

> Which pixels differ from the learned background?

Blob extraction answers:

> Which connected foreground regions might be objects?

The A1 tracker answers:

> Which detection belongs to which persistent track?

Keeping these stages separate gives us a reusable detector-to-tracker contract.

## 2. MOG2 intuition

MOG2 models each pixel with a mixture of Gaussian distributions:

$$
p(x_t)=\sum_{k=1}^{K} w_k \mathcal{N}(x_t;\mu_k,\Sigma_k)
$$

Multiple modes help represent dynamic-but-recurring background behavior better than a single mean.

Important controls:

- history — how much recent evidence influences the model
- variance threshold — how different a sample can be before becoming foreground
- learning rate — how quickly the model adapts
- shadow detection — optional separate shadow labeling

## 3. Foreground cleanup

Raw MOG2 masks often include isolated noise and holes.

The lesson uses:

~~~text
threshold > 200
  |
morphological opening
  |
morphological closing
~~~

Opening removes small speckles. Closing fills small gaps.

The morphology kernel size is configurable with:

~~~text
--morph-kernel
~~~

## 4. Connected components / blobs

Connected-component analysis converts foreground pixels into candidate objects.

For each component the implementation reads:

- area
- x/y
- width/height

Small regions are rejected using:

~~~text
--min-area
~~~

Each remaining blob becomes a generic detection.

## 5. Detection CSV contract

The output schema is exactly what A1 consumes:

~~~text
frame,x,y,w,h,score,class_id
~~~

Example end-to-end flow:

~~~bash
./build/cv9x_background_camshift \
  --video static_camera.mp4 \
  --output-dir build/mog2

./build/cv9x_multi_object_tracking \
  --detections build/mog2/mog2_detections.csv \
  --video static_camera.mp4 \
  --output-dir build/mot
~~~

This gives:

~~~text
video -> MOG2 blobs -> detections -> association -> persistent IDs
~~~

## 6. Why static camera matters

Background subtraction assumes the camera frame itself is approximately stable.

Camera shake or ego-motion can turn much of the image into false foreground.

Moving-camera systems typically need:

- global motion compensation
- image stabilization
- homography compensation
- optical-flow compensation
- detector-based proposals instead of raw background subtraction

## 7. HSV histogram tracking

MeanShift and CamShift use appearance rather than connected-component detections.

The lesson builds a hue histogram from the initial ROI after rejecting low-saturation/dark pixels.

OpenCV hue uses 180 bins for 8-bit HSV, so the target model is:

~~~text
180-bin hue histogram
~~~

## 8. Histogram back-projection

Back-projection creates a single-channel image where each pixel receives a value based on how well its hue matches the target histogram.

High value means the pixel looks like the target color.

Low value means it does not.

The self-test verifies that the target region has a much stronger response than the dark background.

## 9. MeanShift

MeanShift repeatedly moves a fixed-size search window toward the mode of the back-projection distribution.

Conceptually:

~~~text
window
  |
weighted center
  |
shift
  |
repeat
~~~

Its size stays fixed.

That makes it simple, but it cannot naturally follow strong scale change.

## 10. CamShift

CamShift means Continuously Adaptive Mean Shift.

After localization it uses distribution moments to adapt:

- center
- size
- orientation

The deterministic synthetic target grows over time, and the self-test checks that the CamShift window grows too.

## 11. MeanShift vs CamShift

MeanShift:

- fixed window size
- simple
- good when target scale is stable

CamShift:

- adaptive size
- adaptive orientation estimate
- better when the target scale changes
- more sensitive to broad/distracting back-projection responses

Neither one provides multi-object identity management by itself.

## 12. Deterministic synthetic MOG2 demo

Synthetic mode first shows a clean static background so MOG2 can learn it.

Two colored objects then appear and move.

Outputs:

~~~text
01_background_model_input.png
02_foreground_raw.png
03_foreground_clean.png
04_blob_overlay.png
mog2_blob_overlay.avi
mog2_detections.csv
~~~

The detection CSV is directly consumable by A1.

## 13. Deterministic appearance demo

A colored target translates and grows while a differently colored distractor remains in the frame.

Outputs:

~~~text
05_hue_backprojection.png
meanshift_camshift_overlay.avi
appearance_tracking_metrics.csv
~~~

Overlay convention:

~~~text
gray   ground truth
yellow MeanShift
blue   CamShift
~~~

Metrics:

- mean center error
- mean IoU
- final IoU

For persistent multi-object identity, use A1's MOTA/IDF1 evaluation.

## 14. Self-test

~~~bash
./build/cv9x_background_camshift --self-test
~~~

The deterministic tests verify:

1. morphology removes isolated speckles
2. blob extraction recovers the expected foreground box
3. MOG2 learns the static background
4. newly appearing objects are detected
5. blob overlap with truth is high
6. the hue histogram has 180 bins
7. back-projection separates target from background
8. MeanShift maintains bounded center error
9. CamShift maintains bounded center error
10. CamShift preserves useful overlap during scale change
11. CamShift expands its window as the target grows

## 15. Real-video mode

MOG2/blob detection:

~~~bash
./build/cv9x_background_camshift \
  --video input.mp4 \
  --output-dir build/background_tracking
~~~

Add MeanShift/CamShift from an initial first-frame ROI:

~~~bash
./build/cv9x_background_camshift \
  --video input.mp4 \
  --roi 120,80,60,90 \
  --output-dir build/background_tracking
~~~

Useful parameters:

~~~text
--history
--var-threshold
--min-area
--morph-kernel
--learning-rate
--no-shadows
~~~

## 16. Failure cases

### Camera motion

Produces widespread false foreground.

### Sudden illumination change

Can temporarily classify large image regions as foreground.

### Stopped foreground object

May eventually be absorbed into the background model.

### Similar target/background hue

Hurts histogram tracking.

### Blob merge

Two objects touching may become one detection.

### Long occlusion

MeanShift/CamShift can lock onto a distractor.

## 17. Engineering takeaway

A reusable classical perception chain should keep contracts explicit:

~~~text
image acquisition
    |
object proposal / detection
    |
detection list
    |
association + state estimation
    |
persistent tracks
    |
downstream application
~~~

The same A1 association layer can accept MOG2 blobs today and an ML detector later.

## Exercises

1. Sweep MOG2 history.
2. Sweep variance threshold.
3. Compare shadows on/off.
4. Add mask noise and tune morphology.
5. Replace connected components with contours.
6. Add aspect-ratio filtering.
7. Add a same-color distractor.
8. Slowly update the histogram and observe drift.
9. Feed the generated MOG2 detection CSV into A1 and measure MOTA/IDF1.
10. Add global-motion compensation before MOG2.

## Next

A3 moves from the linear constant-velocity Kalman model to an Extended Kalman Filter, including nonlinear turning motion and estimator-consistency checks such as NEES.
