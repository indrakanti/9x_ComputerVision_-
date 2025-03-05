# Laplacian Pyramid and Image Residuals

## Overview
The Laplacian pyramid is a multi-scale image representation that captures image residuals at different levels of resolution. It is built using a Gaussian pyramid but focuses on the **differences (residuals)** between consecutive levels of the Gaussian pyramid. These residuals help in **image compression, blending, and reconstruction** by preserving high-frequency details lost during downsampling.

## Importance of Laplacian Pyramids
- **Image Compression:** Stores only the residuals and reconstructs the original image efficiently.
- **Image Blending:** Enables smooth multi-resolution blending of images.
- **Super-Resolution:** Helps recover finer details in image upscaling.
- **Image Reconstruction:** Allows reconstructing the original image from its smallest form by adding back residuals.

## Construction of a Laplacian Pyramid
A Laplacian pyramid consists of **residual images** that store the difference between each level of a Gaussian pyramid and its upsampled version.

### Steps to Build a Laplacian Pyramid:
1. **Create a Gaussian Pyramid** by progressively blurring and downsampling the original image.
2. **Upsample** each level of the Gaussian pyramid back to its original size.
3. **Compute the Residuals** by subtracting the upsampled image from the corresponding Gaussian level:
   
   \[
   L_k = G_k - \text{Upsample}(G_{k+1})
   \]
   
   where:
   - \( L_k \) is the Laplacian at level \( k \).
   - \( G_k \) is the Gaussian pyramid image at level \( k \).
   - \( \text{Upsample}(G_{k+1}) \) is the upsampled Gaussian image from level \( k+1 \).

4. **Repeat the process** until reaching the smallest level of the pyramid.

## Reconstructing the Original Image
The original image can be reconstructed from the smallest resolution by progressively upsampling and adding back the residuals:

\[
G_k = L_k + \text{Upsample}(G_{k+1})
\]

This iterative process restores lost details, resulting in the **original image**.

## Applications of Laplacian Pyramids
- **Seamless Image Blending** (e.g., multi-exposure image fusion, panorama stitching)
- **Efficient Image Compression** (storing only residuals reduces storage size)
- **Super-Resolution Imaging** (enhancing image details at higher resolutions)
- **Image Reconstruction** (restoring full-scale images from downsampled versions)

## Sample Images Demonstrating Laplacian Pyramids
We will generate a sample image and visualize its Laplacian pyramid representation, showing the image residuals and their role in reconstruction.

