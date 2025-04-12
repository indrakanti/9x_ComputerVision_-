# 🧠 Understanding Scale and Invariance in Feature Detection

---

## 📍 Harris Corner Detector: Strengths & Weaknesses

### ✅ Strengths
- **Good for detecting corners**: It reliably finds high-contrast, corner-like structures in images.
- **Rotation invariant**: Harris uses image gradients, which are not affected by rotation.
- **Relatively efficient**: Computationally lightweight and suitable for real-time systems.

### ❌ Weaknesses
- **Not scale-invariant**: Features found in one resolution might not be detected in another (zoomed-in or zoomed-out).
- **Poor localization at coarse scales**: Without scale adaptation, corner accuracy degrades.
- **Sensitive to thresholding**: Choosing a threshold for the corner response requires manual tuning.
- **Does not compute descriptors**: Harris only detects locations — no native feature descriptors for matching.

---

## 🧭 Characteristic Scale of an Image Filter

**Characteristic scale** is the scale at which a filter responds most strongly to a given image structure.

For example:
- A **blob detector** using a Laplacian of Gaussian (LoG) will give a strong response when the filter size matches the size of the blob in the image.
- A **corner detector** like Harris does not have built-in scale — it works best at the image resolution it was designed for.

> ⬅️ This limitation is addressed in SIFT by building a **scale-space** and searching across multiple filter sizes.

---

## 🔄 Representations and Invariants

| Representation               | Invariant To       | Example Algorithm      |
|------------------------------|--------------------|------------------------|
| Pixel values                 | None               | Raw image              |
| Gradient orientation         | Illumination       | SIFT, HOG              |
| Corner response              | Rotation           | Harris, FAST           |
| DoG extrema in scale-space   | Scale              | SIFT                   |
| BRIEF, ORB                   | Binary intensity   | ORB, BRIEF             |

### Key Takeaways:
- To detect **robust features**, we must choose representations that are **invariant** to the expected transformations (e.g., rotation, scale, illumination).
- For example, **gradient-based representations** help build **illumination invariance**, while **scale-space pyramids** bring **scale invariance**.

---

## 📌 Summary

- **Harris** is good for rotation-invariant corner detection but lacks scale and descriptor robustness.
- **Characteristic scale** of a filter tells you what image structure it is best tuned for.
- Choosing the right representation leads to **robust invariants** — enabling accurate feature detection and matching across different viewpoints, zoom levels, and lighting conditions.

---

Next up: use these concepts to motivate the need for **SIFT**, which builds true **scale-invariant**, **rotation-invariant**, and **descriptor-rich** feature detection pipelines.

# 🔍 ORB Feature Detection and Matching (Scale + Rotation Invariant)

This project demonstrates how to detect and match image features between two images using **ORB** descriptors — a fast and scale-invariant method well-suited for real-time applications.

---

## 📌 Overview

The program:
- Loads two grayscale images
- Detects keypoints using **ORB**
- Computes descriptors
- Matches features using a **Brute-Force Matcher**
- Draws and saves the matched output image

---

## 💡 Why ORB?

**ORB** (Oriented FAST and Rotated BRIEF) combines:
- FAST keypoint detector
- BRIEF descriptor
- Rotation invariance
- Scale invariance (via image pyramids)
- Binary descriptors for fast matching

It is an efficient, open-source alternative to SIFT/SURF and is included in OpenCV by default.

---

## 🧱 Dependencies

- OpenCV 4.x or later
- C++17
- Images placed in `../01_Images/` directory

---

## 📂 Folder Structure

