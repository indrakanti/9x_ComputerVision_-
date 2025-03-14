# Derivative Filter and Edge Detection

## Overview
Edge detection is a crucial step in image processing and computer vision applications. It helps identify significant structural information in an image by detecting regions of rapid intensity change. This can be achieved using **derivative filters**, which compute image gradients to highlight edges.

This document explains the step-by-step process of applying **Gaussian smoothing**, followed by **Sobel and Laplacian filtering**, to detect edges effectively.

## Step 1: Gaussian Smoothing
Before computing image gradients, it's essential to smooth the image to reduce noise. Applying a **Gaussian filter** helps prevent false edges from appearing due to small intensity variations.

#### **Gaussian Kernel (3×3, σ = 1.0):**

$$
G = \frac{1}{16} \begin{bmatrix} 
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1 
\end{bmatrix}
$$

The smoothed image is stored as:
- **`smoothed_image.png`** → The noise-reduced image ready for edge detection.

## Step 2: Understanding the Sobel Filter
The **Sobel filter** is a **first-order derivative filter** that detects edges by computing intensity gradients in an image. Unlike simple derivative filters, the Sobel filter applies **smoothing and differentiation simultaneously**, making it more robust to noise.

### **Concept of Image Gradients**
In an image, edges are regions where the intensity changes significantly. The **gradient** measures this change in intensity at each pixel.

The **gradient at a pixel (x, y)** is given by:

G = $`\sqrt{G_x^2 + G_y^2}`$


where:
- $`( G_x )`$ is the gradient in the **horizontal direction**.
- $`( G_y )`$ is the gradient in the **vertical direction**.
- $`( G  )`$ is the overall **gradient magnitude**, which represents edge strength.

The **direction** of the edge is given by:
$` \theta = \tan^{-1} \left(\frac{G_y}{G_x}\right)`$

This tells us the orientation of the edge.

### **Sobel Operator Kernels**
The Sobel filter uses **two 3×3 convolution kernels** to compute gradients:
1. **Horizontal Gradient Kernel (\( G_x \))**

$$
   G_x =
   \begin{bmatrix}
   -1 & 0 & +1 \\
   -2 & 0 & +2 \\
   -1 & 0 & +1
   \end{bmatrix}
$$
   - This kernel detects **vertical edges** by calculating intensity differences from left to right.

1. **Vertical Gradient Kernel ( $'( G_y \)'$ )**

$$
   G_y =
   \begin{bmatrix}
   -1 & -2 & -1 \\
   0 & 0 & 0 \\
   +1 & +2 & +1
   \end{bmatrix}
$$
   
   - This kernel detects **horizontal edges** by calculating intensity differences from top to bottom.

Each kernel applies a weighted sum of pixel values to enhance directional changes.

### **How Sobel Convolution Works**
For each pixel in the image:
1. Place the **3×3 kernel** over the pixel.
2. Compute the sum of **element-wise multiplications** between the kernel and the pixel values.
3. Store the result as the gradient value at that pixel.
4. Repeat for every pixel in the image.

After obtaining $`( G_x )`$ and $`( G_y )`$, compute:

$` G = \sqrt{G_x^2 + G_y^2} `$
and

$` \theta = \tan^{-1} \left(\frac{G_y}{G_x}\right) `$

### **Combining $`( G_x )`$ and $`( G_y )`$ **
To get the final edge-detected image:
- Compute the **gradient magnitude** at each pixel.
- Apply **thresholding** to remove weak edges.
- The result is a binary or grayscale edge-detected image.

## Summary of Process and Outputs
| Step | Process | Output Image |
|------|---------|--------------|
| 1 | Compute Sobel gradients (X and Y) | `sobel_edges.png` |
| 2 | Compute gradient magnitude | `gradient_magnitude.png` |

## Applications of Derivative Filters
- **Object Detection**: Identifying boundaries in an image.
- **Feature Extraction**: Locating significant image structures.
- **Medical Imaging**: Detecting edges in X-rays and MRI scans.
- **Motion Detection**: Analyzing changes in video sequences.

By following this structured approach, we can achieve **robust edge detection** using derivative filters while minimizing noise interference.

