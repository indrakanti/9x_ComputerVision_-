# Module 18 — Classical Classification: k-NN vs SVM + Feature Pipelines

This lesson makes the supervised classification pipeline explicit.

The pipeline is:

~~~text
image
  |
  v
region / shape extraction
  |
  v
engineered feature vector
  |
  v
training-set normalization
  |
  +-------------------+
  |                   |
  v                   v
k-NN classifier       SVM classifier
  |                   |
  +---------+---------+
            |
            v
       test metrics
            |
            v
confusion / precision / recall
~~~

Both classifiers receive the same feature vectors and the same train/test split so the comparison isolates classifier behavior.

## Learning objectives

By the end of this lesson you should be able to:

- define a classical feature pipeline
- separate feature extraction from classification
- explain why feature scaling matters
- fit normalization on training data only
- recognize preprocessing leakage
- explain k-nearest-neighbor classification
- explain the effect of k
- explain SVM decision boundaries and RBF kernels
- build train/test splits
- compute confusion matrices
- compute accuracy, precision, and recall
- version preprocessing together with the classifier

## 1. Feature vector

Each synthetic shape becomes seven measurements:

~~~text
1. normalized area
2. normalized perimeter
3. circularity
4. aspect ratio
5. extent
6. solidity
7. first Hu invariant, log transformed
~~~

So one sample is:

$$
x \in \mathbb{R}^{7}
$$

This is classical machine learning: the engineer defines the representation, then the classifier learns a decision rule over that representation.

## 2. Classes

The deterministic dataset contains:

~~~text
class 0 = circle
class 1 = rectangle
class 2 = triangle
~~~

Samples vary in size, position, and brightness before thresholding.

The goal is not production shape recognition. The goal is a reproducible supervised-learning pipeline.

## 3. Why scaling matters

The seven dimensions have different numeric ranges.

Distance-based methods such as k-NN can be dominated by one large-scale feature even when that feature is not more important.

RBF-SVM also depends on distance:

$$
K(x_i,x_j)
=
\exp
\left(
-\gamma
\|x_i-x_j\|^2
\right)
$$

So scale matters there too.

## 4. Standardization

For each feature dimension:

$$
\mu_j=
\frac{1}{N}
\sum_i x_{ij}
$$

$$
\sigma_j=
\sqrt{
\frac{1}{N}
\sum_i
(x_{ij}-\mu_j)^2
}
$$

Then:

$$
z_{ij}
=
\frac{x_{ij}-\mu_j}
{\sigma_j}
$$

The training set therefore becomes approximately zero mean and unit standard deviation on every non-constant feature.

## 5. Data leakage rule

Correct:

~~~text
fit mean/std on training features
apply that same mean/std to test features
~~~

Incorrect:

~~~text
fit scaler using train + test
~~~

The second path leaks test information into preprocessing.

Preprocessing is part of the trained model.

## 6. k-nearest neighbors

For a query vector:

1. compute distances to stored training vectors
2. choose the k nearest
3. let their labels vote

Default:

~~~text
k = 5
~~~

Small k creates a more local/noise-sensitive boundary.

Larger k smooths the boundary but can erase small class regions.

k is a validation-selected hyperparameter, not a universal constant.

## 7. k-NN engineering behavior

k-NN has very little traditional fitting.

That means:

~~~text
cheap training
higher prediction cost
memory grows with training set
~~~

This is useful for understanding the difference between lazy and parametric learners.

## 8. SVM

An SVM searches for a decision surface with useful margin.

The previous lesson used a linear SVM.

This lesson uses an RBF kernel to introduce nonlinear decision boundaries.

Teaching defaults:

~~~text
C = 4.0
gamma = 0.5
~~~

These are not universal optimal settings.

## 9. Train/test separation

Default dataset:

~~~text
90 training samples per class
30 test samples per class
~~~

Total:

~~~text
270 training
90 testing
~~~

The test set is not used to fit:

- normalization
- k-NN stored training set
- SVM parameters

## 10. Confusion matrix

Rows are true classes.

Columns are predicted classes.

Diagonal entries are correct predictions.

Off-diagonal entries expose specific class confusions.

## 11. Accuracy

$$
accuracy=
\frac{correct}{all}
$$

Useful, but incomplete.

## 12. Precision

For class c:

$$
precision_c=
\frac{TP_c}
{TP_c+FP_c}
$$

Interpretation:

> When the model predicts class c, how often is it right?

## 13. Recall

For class c:

$$
recall_c=
\frac{TP_c}
{TP_c+FN_c}
$$

Interpretation:

> Of the real class-c samples, how many did the model recover?

## 14. Fair classifier comparison

This lesson holds constant:

- dataset
- feature extraction
- split
- scaling

and changes only:

~~~text
classifier
~~~

That makes the comparison interpretable.

## 15. Feature-space visualization

The executable writes:

~~~text
02_feature_scatter.png
~~~

It plots standardized:

~~~text
x = circularity
y = aspect ratio
~~~

This is only a 2-D projection of the seven-dimensional feature space.

Overlap in this plot does not imply the full feature vectors are inseparable.

## 16. Outputs

~~~text
01_class_examples.png
02_feature_scatter.png
03_knn_confusion.png
04_svm_confusion.png
test_predictions.csv
~~~

The CSV contains:

- true label
- k-NN prediction
- SVM prediction
- seven raw features
- seven standardized features

## 17. Deterministic self-test

Run:

~~~bash
./build/cv9x_classical_classification --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- expected train dimensions
- expected test dimensions
- standardized training means are near zero
- k-NN trains
- SVM trains
- k-NN test accuracy is at least 95 percent
- SVM test accuracy is at least 95 percent
- both confusion matrices account for every test sample
- every class has at least 90 percent recall for both classifiers

These thresholds apply only to this controlled synthetic task.

## 18. Why synthetic data helps

Synthetic data provides:

- deterministic labels
- reproducible CI
- controlled variation
- no external download
- easy failure injection

But high synthetic accuracy does not prove real-world transfer.

## 19. Model contract

A classical classifier is more than a model file.

The contract includes:

~~~text
segmentation method
feature definition
feature ordering
feature units
normalization mean
normalization stddev
classifier type
hyperparameters
label mapping
model version
training-data version
~~~

Changing feature order while keeping the same model silently breaks correctness.

## 20. k-NN vs SVM behavior

### k-NN

Strengths:

- simple
- interpretable neighbors
- minimal fitting
- easy to update by changing stored examples

Costs:

- prediction cost grows with data
- memory grows with data
- distance metric and scaling matter strongly

### SVM

Strengths:

- compact learned decision representation
- strong margin-based classifier
- kernels support nonlinear boundaries

Costs:

- training and hyperparameter selection are required
- preprocessing consistency is critical
- kernel models can grow with support-vector count

There is no universal winner; system constraints and validation evidence decide.

## Common failure cases

### Fit scaling on train and test together

Leakage.

### Fit a separate scaler on test/inference data

The model sees a different coordinate system.

### Forget inference normalization

Distribution mismatch.

### Reorder feature columns

The model receives the wrong semantics.

### Tune k, C, or gamma on the test set

Model selection belongs to validation data.

### Report only accuracy

Inspect confusion, precision, recall, and representative failures.

### Treat synthetic accuracy as real-world proof

It is not.

## Engineering notes

Persist or version:

~~~text
feature schema
feature order
normalization mean/std
class-label mapping
classifier hyperparameters
training dataset
validation metrics
software version
numeric precision
~~~

At runtime, verify model and feature-producer compatibility.

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_classical_classification
~~~

## Run

~~~bash
./build/cv9x_classical_classification \
  --output-dir build/classical_classification \
  --train-per-class 90 \
  --test-per-class 30 \
  --knn-k 5 \
  --seed 42
~~~

## Exercises

1. Run k-NN without standardization.
2. Sweep k from 1 to 25.
3. Replace RBF SVM with linear SVM.
4. Sweep C and gamma using a validation split.
5. Remove one feature at a time.
6. Add feature noise.
7. Create class imbalance.
8. Add a fourth overlapping class.
9. Persist scaler parameters and models, then reload them.
10. Replace shape features with HOG or SIFT-derived features.

## Next lesson

Module 19 covers classical tracking:

- template matching
- correlation tracking
- Kalman-filter basics
- prediction vs measurement
- track lifecycle

That completes the original classical Recognition & Tracking section before deciding the separate ML/deep-vision repository boundary.
