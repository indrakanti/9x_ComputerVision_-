# Module 17 — HOG + Linear SVM Object Detection

This lesson moves from explicit geometric shape descriptors into appearance-based object detection.

The pipeline is:

~~~text
image window
   |
   v
image gradients
   |
   v
8x8 cells
   |
   v
orientation histograms
   |
   v
2x2-cell block normalization
   |
   v
3780-D HOG descriptor
   |
   v
linear SVM
   |
   v
sliding-window classification
   |
   v
multi-scale detections + NMS
~~~

The HOG descriptor is implemented explicitly in C++. OpenCV is used for the linear SVM training API.

## Learning objectives

By the end of this lesson you should be able to:

- explain why local gradient orientation describes object appearance
- distinguish gradient magnitude from gradient orientation
- explain unsigned 0–180 degree HOG bins
- build per-cell orientation histograms
- understand orientation-bin interpolation
- explain overlapping block normalization
- implement L2-Hys normalization
- derive the classical 3780-D descriptor length
- explain a linear SVM decision boundary
- train/test a classical feature classifier
- turn a fixed-window classifier into a detector using sliding windows
- build an image pyramid for scale variation
- apply non-maximum suppression
- understand the limitations of classical HOG detectors

## 1. From shape to appearance

Contours work well when a clean foreground mask already exists.

HOG instead asks:

> What spatial pattern of local edge orientations appears inside this candidate window?

This lets us classify raw image windows without requiring a prior contour.

## 2. Gradients

For each pixel:

$$
G_x,\quad G_y
$$

Magnitude:

$$
m=\sqrt{G_x^2+G_y^2}
$$

Orientation:

$$
\theta=\operatorname{atan2}(G_y,G_x)
$$

This lesson uses unsigned orientation:

$$
0^\circ \le \theta < 180^\circ
$$

Opposite gradient signs can therefore represent the same geometric edge direction.

## 3. Cells and bins

The canonical window is:

~~~text
64 x 128 pixels
~~~

Cell size:

~~~text
8 x 8 pixels
~~~

Therefore:

~~~text
8 cells horizontally
16 cells vertically
~~~

Each cell accumulates a 9-bin orientation histogram.

With 9 bins across 180 degrees:

$$
\Delta\theta=20^\circ
$$

Each pixel contributes its gradient magnitude to neighboring orientation bins using linear interpolation.

## 4. Blocks and normalization

Cells are grouped into overlapping:

~~~text
2 x 2 cell blocks
~~~

For block vector \(v\), L2-Hys does:

$$
v'=
\frac{v}
{\sqrt{\|v\|_2^2+\epsilon^2}}
$$

clip:

$$
v'_i=\min(v'_i,0.2)
$$

then normalize again.

This reduces sensitivity to local illumination/contrast changes and prevents a few strong gradients from dominating.

## 5. Why 3780 dimensions?

Window:

~~~text
64 x 128
~~~

Cells:

~~~text
8 x 16
~~~

Overlapping 2x2-cell blocks:

~~~text
7 x 15 block positions
~~~

Values per block:

~~~text
2 x 2 x 9 = 36
~~~

Therefore:

$$
7\times15\times36=3780
$$

The self-test checks this exact descriptor length.

## 6. HOG visualization

The executable writes:

~~~text
03_hog_cells.png
~~~

Dominant orientation bins are drawn within each cell.

The visualization uses raw cell histograms for intuition. The actual classifier uses the flattened block-normalized descriptor.

## 7. Linear SVM

Every training window becomes one point in 3780-dimensional feature space.

Labels:

~~~text
+1 = target
-1 = background / non-target
~~~

A linear SVM learns a separating hyperplane:

$$
w^Tx+b=0
$$

This is the classical pattern:

~~~text
hand-engineered feature representation
+
simple learned classifier
~~~

## 8. Synthetic training set

To keep CI deterministic and dataset-free, the lesson creates synthetic data.

Positive samples contain a simplified upright human-like silhouette with small position/thickness jitter.

Negative samples include:

- horizontal rectangles
- circles
- horizontal stripes
- vertical stripes
- random block patterns

This is not a real pedestrian detector. It is a reproducible teaching dataset for the complete pipeline.

## 9. Holdout testing

Training accuracy is not enough.

The lesson creates a separate holdout set from another random seed and reports holdout accuracy.

The self-test requires at least 95% on this deliberately controlled synthetic task.

Real projects need stronger evaluation:

- independent train/validation/test splits
- precision/recall
- PR/ROC curves
- dataset provenance
- hard-negative mining

## 10. Sliding-window detection

The SVM classifies one 64x128 window.

To detect in a larger image:

~~~text
for every window location:
    crop 64x128
    compute HOG
    classify with SVM
~~~

Default stride:

~~~text
8 pixels
~~~

Every positive window becomes a candidate detection.

## 11. Scale pyramid

The detector window is fixed-size.

To detect different object sizes, the image is repeatedly resized:

~~~text
scale 1.00
scale 1.20
scale 1.44
...
~~~

The same 64x128 detector is scanned at every level and detections are mapped back into original-image coordinates.

## 12. Non-maximum suppression

Many nearby windows can fire on one target.

NMS uses intersection-over-union:

$$
IoU=
\frac{|A\cap B|}
{|A\cup B|}
$$

The lesson suppresses heavily overlapping boxes after ranking positive SVM windows by absolute hyperplane margin.

OpenCV raw SVM margin sign may depend on internal label orientation, so normal SVM prediction decides positive/negative first.

## 13. Deterministic detection scene

Without input, the executable generates a scene containing:

- one synthetic target
- one rectangle distractor
- one circular distractor
- background line texture

The target lies on the scanning grid.

The self-test requires the best retained detection to satisfy:

$$
IoU\ge0.45
$$

with the known target box.

## 14. Outputs

~~~text
01_positive_sample.png
02_negative_sample.png
03_hog_cells.png
04_detection_input.png
05_hog_svm_detections.png
~~~

The console reports:

- descriptor configuration
- descriptor length
- training sample counts
- holdout accuracy
- retained detection count
- best synthetic IoU

## 15. Deterministic self-test

Run:

~~~bash
./build/cv9x_hog_svm --self-test
ctest --test-dir build --output-on-failure
~~~

It verifies:

- HOG descriptor length is 3780
- a constant image produces zero HOG
- 0-degree orientation voting
- 90-degree interpolated voting
- L2-Hys output has unit norm
- linear SVM training completes
- holdout accuracy is at least 95%
- held-out positive/negative samples classify correctly
- sliding-window detection localizes the synthetic target

## 16. Real-image mode

Run:

~~~bash
./build/cv9x_hog_svm \
  --input scene.png \
  --output-dir build/hog_svm
~~~

Important limitation:

The model is still trained on the synthetic teaching target. It should not be expected to detect real pedestrians.

A real detector requires representative real-world data and hard-negative mining.

## 17. Hard-negative mining

A classical detector often improves through:

~~~text
train
 |
run on negative images
 |
collect false-positive windows
 |
add as hard negatives
 |
retrain
~~~

This is an early example of iterative data-centric model development.

## 18. HOG vs contours

Contours describe object boundary geometry after segmentation.

HOG describes local edge-orientation layout directly inside an image window.

They solve different stages of a perception pipeline.

## 19. HOG vs SIFT

Both use gradient orientation histograms.

SIFT:

- local keypoint descriptor
- designed for matching
- orientation-normalized
- compact local neighborhood

HOG:

- dense spatial grid
- preserves coarse object layout
- designed for window-level appearance classification

## 20. Limitations

Classical HOG detectors struggle with:

- large pose variation
- unusual viewpoints
- strong occlusion
- deformable appearance
- small targets
- clutter and domain shift

Runtime also grows with the number of:

~~~text
locations x scales x descriptor computations
~~~

This is one reason later learned detectors became dominant.

## Engineering notes

A production classical detector should version:

~~~text
model
training data
HOG configuration
window size
cell size
block size
orientation bins
normalization
stride
scale pyramid
classification threshold
NMS threshold
preprocessing
~~~

The HOG configuration is part of the model contract. Changing it invalidates the learned SVM weights.

## Build

This lesson adds OpenCV's ml module.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_hog_svm
~~~

## Synthetic demo

~~~bash
./build/cv9x_hog_svm \
  --output-dir build/hog_svm \
  --positives 80 \
  --negatives 80 \
  --test-samples 20 \
  --stride 8 \
  --scale-step 1.20 \
  --max-scales 5
~~~

## Exercises

1. Disable block normalization and measure accuracy.
2. Change orientation bins and recompute descriptor size.
3. Change cell size and explain model incompatibility.
4. Add brightness/contrast augmentation.
5. Add harder negatives.
6. Implement hard-negative mining.
7. Compare manual HOG with OpenCV HOGDescriptor.
8. Plot SVM margin distributions.
9. Measure stride vs latency/localization.
10. Replace the synthetic dataset with a real pedestrian dataset.

## Next lesson

The roadmap next covers classical classification with k-NN, SVM, and explicit feature pipelines.
