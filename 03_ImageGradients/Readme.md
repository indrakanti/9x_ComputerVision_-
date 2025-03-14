# Derivative Filter and Image Gradients

## Overview
Edges in an image correspond to regions of rapid intensity change. Detecting these edges is a fundamental task in computer vision and image processing. One of the most effective ways to locate these changes is through **image gradients**, which measure the rate of intensity variation.

A **derivative filter** is a simple yet powerful tool that computes the gradient of an image by approximating the derivative in one or more directions. This forms the foundation for edge detection techniques such as the **Sobel filter, Prewitt filter, and Canny edge detection**.

## Importance of Smoothing Before Derivative Filtering
Applying a smoothing filter before computing derivatives helps suppress noise and improve edge detection accuracy. Without smoothing, small intensity variations may create false edges. A common approach is to use a **Gaussian filter** before applying derivative filters.

### **Step 1: Apply Gaussian Smoothing**
A Gaussian filter smooths the image by averaging pixel intensities, reducing noise before computing gradients.

#### **Gaussian Kernel (3×3, σ = 1.0):**
$$
G = \frac{1}{16} \begin{bmatrix} 
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1 
\end{bmatrix}
$$

This step ensures that only significant edges remain in the processed image.

### **Step 2: Apply Derivative Filtering**
## Understanding Image Gradients
An image gradient measures the rate of change in pixel intensities across an image. The gradient is computed using partial derivatives in the **x-direction (horizontal)** and **y-direction (vertical)**:

$$
G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}
$$

where:
- \( $$ G_x $$ \) represents changes in intensity along the horizontal axis.
- \( $$ G_y $$ \) represents changes in intensity along the vertical axis.
- \( I \) is the input image.

The magnitude of the gradient is computed as:
$$
G = \sqrt {G_x^2 + G_y^2}
$$
This measures the strength of edges in the image.

## Derivative Filters
### 1. First-Order Derivative Filters
These filters approximate the first derivative of the image by computing intensity differences between neighboring pixels. Common first-order derivative filters include:

#### a) **Roberts Cross Operator**
\[
G_x = \begin{bmatrix} +1 & 0 \\ 0 & -1 \end{bmatrix}, \quad G_y = \begin{bmatrix} 0 & +1 \\ -1 & 0 \end{bmatrix}
\]

#### b) **Sobel Filter: A Derivative Filter with Spatial Smoothing**
The Sobel filter enhances edge detection by incorporating smoothing and directional sensitivity, making it highly effective for detecting edges in various orientations.

\[
G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}
\]

The Sobel filter applies a weighted difference to pixels, emphasizing edge strength while reducing sensitivity to noise.

### 2. Second-Order Derivative Filters
These filters compute the second derivative of an image, which highlights changes in gradient intensity. A common second-order filter is:

#### **Laplacian Operator**
\[
L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}
\]
This filter detects edges regardless of direction by capturing areas where the gradient changes abruptly.

## Applications of Derivative Filters
- **Edge Detection:** Used in object detection and image segmentation.
- **Feature Extraction:** Helps identify important image structures.
- **Optical Flow Estimation:** Computes motion in video sequences.

## Sample Images Demonstrating Smoothing and Derivative Filtering
We will generate a sample image and apply **Gaussian smoothing first**, followed by derivative filters, to visualize how they improve edge detection results.

