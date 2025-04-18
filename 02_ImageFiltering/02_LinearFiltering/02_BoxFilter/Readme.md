# Box Filter in Image Processing

## Overview
A box filter is a simple spatial filtering technique used in image processing to blur an image, reduce noise, or smooth edges. It works by averaging the pixel values in a defined neighborhood around each pixel, effectively replacing each pixel with the mean of its surrounding pixels. This operation helps remove high-frequency components and smoothens the image.

## Mathematical Representation
A box filter is defined using a convolution operation with a kernel, where each value in the kernel is equal and sums to 1:

$$
G(x, y) = \frac{1}{N} \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x+i, y+j)
$$

where:
- \( G(x, y) \) is the output pixel value at position \( (x, y) \).
- \( I(x+i, y+j) \) represents the input pixel values within the filter window.
- \( N \) is the total number of pixels in the filter window (e.g., for a 3x3 filter, \( N=9 \)).
- \( k \) determines the size of the filter window (for a 3x3 filter, \( k=1 \)).

## Effects of Box Filtering
- **Blurring:** Reduces fine details by averaging neighboring pixel values.
- **Noise Reduction:** Helps suppress random variations in pixel intensities.
- **Edge Softening:** Can reduce sharp transitions in an image, leading to smoother edges.

Example:  This is a sample synthetic image.
![Sample input image](../01_Images/original_synthetic.png)


## Example of Box Filter Kernels
### 3x3 Box Filter:
$$
\frac{1}{9} \begin{bmatrix} 
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 
\end{bmatrix}
$$

Example: ![3x3 box filter sample](../01_Images/box_filter_3x3.png)

### 5x5 Box Filter:
$$
\frac{1}{25} \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 \\ 
1 & 1 & 1 & 1 & 1 
\end{bmatrix}
$$

Example: ![5x5 box filter sample](../01_Images/box_filter_5x5.png)

### 7x7 Box Filter:
$$
\frac{1}{49} \begin{bmatrix} 
1 & 1 & 1 & 1 & 1 & 1 & 1  \\
1 & 1 & 1 & 1 & 1 & 1 & 1  \\ 
1 & 1 & 1 & 1 & 1 & 1 & 1  \\ 
1 & 1 & 1 & 1 & 1 & 1 & 1  \\ 
1 & 1 & 1 & 1 & 1 & 1 & 1  \\ 
1 & 1 & 1 & 1 & 1 & 1 & 1  \\ 
1 & 1 & 1 & 1 & 1 & 1 & 1  
\end{bmatrix}
$$

Example: ![5x5 box filter sample](../01_Images/box_filter_7x7.png)


