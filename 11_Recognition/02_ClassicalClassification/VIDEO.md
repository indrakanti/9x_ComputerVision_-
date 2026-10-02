# Video: k-NN vs SVM — Classical Classification Pipelines in C++

YouTube: TBD

Suggested title:

**k-NN vs SVM in C++ — Feature Scaling, Confusion Matrices & Classical Classification**

Suggested thumbnail:

**SAME FEATURES. DIFFERENT CLASSIFIERS.**

Target duration: **30–36 minutes**

## Learning objectives

Viewers should understand:

- explicit feature vectors
- training/test separation
- z-score feature scaling
- preprocessing leakage
- k-NN distance-based classification
- RBF-SVM classification
- confusion matrices
- precision and recall
- why preprocessing belongs to the model contract

## Chapters

~~~text
00:00 From object detection to general classification
02:00 Build a 7-D feature vector
05:00 Why feature scaling matters
09:00 Train-only normalization
12:00 k-nearest neighbors
16:00 What changing k does
19:00 RBF SVM
23:00 Train/test split and leakage
26:00 Confusion matrix
29:00 Accuracy, precision and recall
32:00 Compare k-NN and SVM
34:00 Production model contract
35:30 Next: classical tracking
~~~

## Script outline

### 00:00 — Same features, different classifier

Show one circle, rectangle, and triangle.

Then show the seven extracted features.

Say:

> In the last lessons we focused on descriptors and one detector. Today we isolate the classifier itself.

### 02:00 — Feature vector

List:

~~~text
area
perimeter
circularity
aspect ratio
extent
solidity
Hu1
~~~

Explain that the classifier never sees the original image.

It sees only these numbers.

### 05:00 — Scaling

Show two dimensions with very different numeric ranges.

Explain why Euclidean distance becomes misleading when one dimension dominates numerically.

### 09:00 — No leakage

Draw:

~~~text
TRAIN -> fit mean/std
TEST  -> use train mean/std
~~~

Cross out:

~~~text
TRAIN + TEST -> fit scaler
~~~

Explain that preprocessing statistics are learned parameters too.

### 12:00 — k-NN

Put one test point into feature space.

Find its five nearest training samples.

Let them vote.

Discuss memory and prediction-cost behavior.

### 16:00 — k

Compare:

~~~text
k = 1
k = 5
large k
~~~

Explain local sensitivity vs smoothing.

### 19:00 — RBF SVM

Start from the linear SVM from the HOG lesson.

Then introduce:

$$
K(x_i,x_j)
=
\exp(-\gamma\|x_i-x_j\|^2)
$$

Explain C and gamma conceptually.

### 23:00 — Train/test split

Show deterministic but separate random seeds.

Stress:

> Test data evaluates the final pipeline. It is not a tuning resource.

### 26:00 — Confusion matrix

Rows = true.

Columns = predicted.

Show diagonal and off-diagonal cells.

### 29:00 — Metrics

Write:

$$
accuracy=correct/all
$$

$$
precision=TP/(TP+FP)
$$

$$
recall=TP/(TP+FN)
$$

Explain why class-specific metrics matter.

### 32:00 — Demo

Run:

~~~bash
./build/cv9x_classical_classification \
  --output-dir build/classical_classification
~~~

Show:

- class examples
- feature scatter
- k-NN confusion
- SVM confusion
- prediction CSV

### 34:00 — Engineering contract

Show that a deployed model needs:

~~~text
feature order
mean/std
labels
hyperparameters
model version
~~~

not merely a classifier file.

### 35:30 — Next

Preview template matching, correlation, Kalman prediction, and track lifecycle.

## Commands demonstrated

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_classical_classification

./build/cv9x_classical_classification \
  --output-dir build/classical_classification \
  --train-per-class 90 \
  --test-per-class 30 \
  --knn-k 5 \
  --seed 42

./build/cv9x_classical_classification --self-test
ctest --test-dir build --output-on-failure
~~~

## Short idea

**The Easiest Way to Leak Your Test Set**

Show feature normalization.

Wrong: calculate mean/std using all data.

Correct: fit only on training data, then reuse those exact parameters for validation/test/inference.

## YouTube description

This lesson builds a complete classical classification pipeline in C++.

We extract explicit shape features, standardize them using training data only, compare k-nearest neighbors with an RBF SVM, and evaluate both classifiers using confusion matrices, accuracy, precision, and recall.

Source code + lesson notes:
https://github.com/indrakanti/9x_ComputerVision_-/tree/main/11_Recognition/02_ClassicalClassification

Full course:
https://github.com/indrakanti/9x_ComputerVision_-

#ComputerVision #OpenCV #CPP #KNN #SVM #MachineLearning #Linux
