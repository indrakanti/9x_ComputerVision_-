# Partial Derivatives and Edge Detection

## Overview
Partial derivatives are essential in understanding how an image changes in different directions. In image processing, they are the foundation of **gradient-based edge detection**. The **image gradient** provides crucial information about intensity transitions, enabling us to detect edges, corners, and boundaries.

---

## What Are Partial Derivatives?
A **partial derivative** measures the rate of change of intensity along one specific direction:

- \( \frac{\partial I}{\partial x} \): Change in intensity along the **horizontal (x)** axis.
- \( \frac{\partial I}{\partial y} \): Change in intensity along the **vertical (y)** axis.

These derivatives capture local variations and are computed using convolution kernels (like **Sobel** or **Prewitt** filters).

---

## Gradient and Edge Detection
The **image gradient** is a vector that combines both partial derivatives:

\[
\nabla I = \left[ \frac{\partial I}{\partial x}, \frac{\partial I}{\partial y} \right]
\]

- **Gradient Magnitude** (Edge Strength):
\[
|\nabla I| = \sqrt{I_x^2 + I_y^2}
\]

- **Gradient Direction** (Edge Orientation):
\[
\theta = \tan^{-1}\left(\frac{I_y}{I_x}\right)
\]

- **High gradient magnitudes** correspond to **edges** in the image.
- The **direction** tells us the angle of the edge.

---

## Derivative Operators
The most common way to approximate partial derivatives in an image is using convolution kernels:

### Sobel Kernels:
- \( G_x \):
\[
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
\]

- \( G_y \):
\[
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
\]

These are applied to the image to extract \( I_x \) and \( I_y \).

---

## Application in Edge Detection
By combining \( I_x \) and \( I_y \), we generate:
- A gradient magnitude image showing **where edges occur**.
- A gradient direction image showing **how those edges are oriented**.

These results are essential for subsequent steps like **corner detection** or **feature extraction**.

---

## Next Step
Use the C++ implementation to:
- Load a grayscale image
- Apply Sobel filters to compute \( I_x \), \( I_y \)
- Calculate gradient magnitude and orientation
- Visualize the result

Let's move to the C++ implementation next. ✅

