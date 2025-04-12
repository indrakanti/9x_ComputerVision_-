# 🔍 SIFT – Stage 1: Keypoint Detection using Scale-Space Extrema

## 📌 Overview

This stage introduces the first part of the **SIFT (Scale-Invariant Feature Transform)** pipeline — detecting stable **keypoints** that are invariant to scale and rotation. The method uses the **Difference of Gaussian (DoG)** pyramid to find candidate keypoints in an image.

---

## 🧠 Why Scale-Invariance?

Traditional detectors like Harris are not scale-invariant. SIFT solves this by detecting features in a **scale-space**, ensuring that keypoints are robust to:
- Zoom
- Viewpoint changes
- Multi-scale textures

---

## 🌐 What is Scale-Space?

The **scale-space** is a core concept in computer vision that allows us to analyze images at different levels of detail. In the context of SIFT, it enables **detecting features that remain consistent even when the object appears larger or smaller**.

### 📚 Formal Definition

Scale-space is a continuous function of an image \( I(x, y) \) over spatial coordinates and a **scale parameter** \( \sigma \), which controls the level of Gaussian blur applied.

\[
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
\]

Where:
- \( * \) denotes convolution
- \( G(x, y, \sigma) \) is a 2D Gaussian kernel:
\[
G(x, y, \sigma) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
\]

### 🧠 Intuition

- **Small σ (sharper images)** capture fine textures and noise
- **Large σ (more blurred images)** capture coarse structures and global features
- By analyzing the image across **multiple σ**, we can identify structures (blobs, corners) that are meaningful **regardless of image resolution**

---

## 🧱 Building the Scale-Space in SIFT

SIFT constructs the scale-space as a **Gaussian Pyramid**, where:
- Each **octave** halves the resolution (downsampling)
- Each octave contains multiple **blurred layers** (images at increasing σ)
- SIFT typically uses 3-4 octaves and 4-5 scales per octave

Example:

---

## 📐 Step-by-Step Pipeline

### 1. 🌀 Build Gaussian Pyramid
- Start with the input image
- Apply multiple **Gaussian blurs** with increasing σ (scale)
- Downsample the image between pyramid levels (octaves)

### 2. ➖ Compute Difference-of-Gaussian (DoG)
- Subtract consecutive blurred images:
  \[
  DoG(i) = G(i+1) - G(i)
  \]
- DoG approximates the **Laplacian of Gaussian (LoG)** but is much faster

### 3. 🔎 Detect Scale-Space Extrema
- For each pixel in the DoG pyramid:
  - Compare it with **8 neighbors** in the same scale
  - Compare it with **9 neighbors** in the scale above
  - Compare it with **9 neighbors** in the scale below
- If it is a **local minimum or maximum** → it's a keypoint candidate

### 4. 📤 Output
- Keep keypoint locations and the scale (σ) where they were found
- These are passed to the next stage (filtering, orientation)

---

## 📈 Visual Representation

