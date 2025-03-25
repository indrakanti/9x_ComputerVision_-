# Image Blending using Gaussian and Laplacian Pyramids

## Overview
In this exercise, you will learn how to perform image blending using **pyramid-based techniques**. A powerful example is blending a **sunset** and a **cityscape** into one seamless composite. You will create two aligned images, slice them, and use **Gaussian and Laplacian pyramids** to blend them together.

---

## Task Summary
1. **Capture or gather two visually distinct but complementary images** (e.g., `sunset.png`, `cityscape.png`).
2. **Align the images** so that important features (like the skyline) are well-positioned and horizontally aligned.
3. **Slice the images vertically** — left half of one image and right half of the other.
4. **Blend the images** using Laplacian and Gaussian pyramid techniques.
5. **Export the final result** and share it.

---

## Step-by-Step Instructions

### 1. Image Preparation
- Obtain two high-quality images: `sunset.png` and `cityscape.png`.
- Make sure the scenes are:
  - Similar in **dimensions and composition** (e.g., horizon line).
  - Taken from **similar perspectives**.
- Resize and crop the images so they are of the **same dimension** (e.g., 512×512 pixels).

### 2. Align the Images
- Use basic image processing steps to:
  - Crop unnecessary parts.
  - Resize both images to the **same width and height**.
  - Optionally use **affine** or **perspective transformations** to improve alignment.

### 3. Image Slicing
- Slice both images vertically down the center:
  - From `sunset.png`, keep the **left half**.
  - From `cityscape.png`, keep the **right half**.

```python
cols = image.shape[1] // 2
left_half = sunset[:, :cols]
right_half = cityscape[:, cols:]
```

- Combine the halves using pyramid blending instead of hard concatenation.

### 4. Blending with Pyramids

#### a. Gaussian Pyramid
- Build Gaussian pyramids for both images and the mask.
- Use `cv2.pyrDown()` to create a pyramid of **progressively smaller images**.

#### b. Laplacian Pyramid
- Compute Laplacian pyramids using differences between successive levels of the Gaussian pyramid.
- These pyramids represent **image detail** at different scales.

#### c. Blend Pyramids
- At each level of the Laplacian pyramid, blend corresponding halves:
  - Left of sunset with right of cityscape.

#### d. Collapse the Pyramid
- Reconstruct the final blended image by adding the levels from the Laplacian pyramid back up:
  - Use `cv2.pyrUp()` to upscale and `cv2.add()` to combine.

### 5. Export the Blended Image
- Save the result as `blended_sunset_cityscape.png`.
- Optionally visualize intermediate pyramid levels for better insight.

---

## Tips for Better Blending
- Try different numbers of pyramid levels (typically 5 or 6).
- Adjust the **sigma (𝜎)** value when applying Gaussian blur to control blending smoothness.
- Use a soft mask (e.g., a gradient) instead of a hard binary mask to improve the transition between halves.

---

## Submission Instructions
- Export and save your final image as `blended_sunset_cityscape.png`.
- Share your result in the discussion forum.
- Check out other blends created by your peers and give feedback!

---

## Example Result
A successful blend should:
- Seamlessly merge the sunset and cityscape into a unified panorama.
- Avoid visible seams or mismatched lighting.
- Look like a naturally blended landscape!

Happy blending! 🌇🏙️

