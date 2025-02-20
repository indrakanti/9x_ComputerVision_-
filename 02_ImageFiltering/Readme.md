# Image Filtering in Computer Vision

## Introduction
Image filtering is a fundamental technique in image processing and computer vision. Filters operate directly on pixel values to achieve various effects such as smoothing, sharpening, and edge detection. These operations are essential for preprocessing images before high-level tasks like object detection, segmentation, or feature extraction.

## Types of Filters
Filters can be broadly categorized into:

### 1. **Linear Filters**
Linear filters modify pixel values by computing a weighted sum of neighboring pixels using a convolution operation. Examples include:
- **Box Blur (Average Filter)** – Averages pixel values in a region to reduce noise and smooth the image.
- **Gaussian Blur** – Applies a weighted averaging based on a Gaussian function to preserve edges while reducing noise.
- **Sharpening Filter** – Enhances edges and fine details by emphasizing intensity changes.
- **Edge Detection Filters (Sobel, Prewitt, Roberts)** – Compute gradients in the image to highlight edges.

### 2. **Non-Linear Filters**
Non-linear filters do not use simple weighted sums but instead apply rules based on pixel values. Examples include:
- **Median Filter** – Replaces each pixel with the median value of its neighborhood to remove salt-and-pepper noise.
- **Bilateral Filter** – Preserves edges while smoothing by considering both spatial distance and intensity difference.
- **Adaptive Filters** – Modify behavior based on local image characteristics (e.g., Adaptive Gaussian filtering).

### 3. **Frequency Domain Filters**
Instead of modifying pixel values directly, these filters transform the image into the frequency domain and apply modifications:
- **Low-Pass Filters (LPF)** – Remove high-frequency details to smooth images (e.g., Gaussian LPF).
- **High-Pass Filters (HPF)** – Enhance high-frequency details like edges and textures.
- **Band-Pass Filters** – Retain specific frequency ranges while removing others.

## How Filters Work
Filters typically work using a **kernel (convolution mask)**, which is a small matrix that slides over the image and modifies pixel values. The operation can be represented as:
\[
I'(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x+i, y+j) K(i, j)
\]
where:
- \( I(x, y) \) is the original pixel value,
- \( K(i, j) \) is the kernel,
- \( I'(x, y) \) is the filtered pixel value,
- \( k \) is the kernel radius.

## Applications of Image Filtering
- **Noise Reduction** – Removing unwanted artifacts before further processing.
- **Edge Detection** – Identifying object boundaries.
- **Feature Enhancement** – Improving contrast and details.
- **Image Restoration** – Correcting degradation due to motion blur or noise.

## Next Steps
In the following sections, we will implement different types of filters in Python using NumPy and OpenCV. Stay tuned for detailed code examples!

