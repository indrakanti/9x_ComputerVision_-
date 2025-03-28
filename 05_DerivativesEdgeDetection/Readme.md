# Harris Corner Detection Project Overview

This repository walks through the complete pipeline of **Harris Corner Detection**, broken down into modular and progressive stages. Each folder corresponds to a specific stage of the algorithm, complete with source code, outputs, and explanations.

---

## 📁 Folder Structure and Contents

### `01_Images`
- 📌 Contains input test images and output reference images.
- 💡 All stages read and write images to/from this folder.
- 📝 Diagrams, animations, and markdown visualizations use content from here.

---

### `02_EdgeDetectionPartialDerivates`
- 📷 Computes first-order derivatives (image gradients).
- ✅ Visualizes gradient direction and strength.
- ⚙️ Implements partial derivative filters like Sobel.

---

### `03_HarrisCorner`
- 🧠 Computes the covariance matrix (structure tensor).
- ✅ Visualizes second-moment components.
- 📊 Prepares gradient products used for later stages.

---

### `04_HarrisCornerTaylorseriers`
- 📐 Demonstrates the **Taylor Series** approximation.
- ✅ Approximates shifted image patches using gradients.
- 📷 Visualizes the concept of tangent planes.

---

### `05_HarrisEigen`
- 🔍 Computes **eigenvalues** of the structure tensor.
- 🧮 Interprets eigenvalues to describe patch variation.
- 📈 Connects eigenvalues to quadratic surface shape.

---

### `06_OptionalClassifyImageRegions`
- 🧪 Uses eigenvalues to classify regions as:
  - Flat (low variation)
  - Edge (one-direction variation)
  - Corner (multi-directional variation)
- 🎯 Produces annotated image with region types.

---

### `07_HarrisResponseFinal`
- 🚀 Implements final Harris Corner Detection using the **Harris response function**:
  
  \[ R = \text{det}(M) - k \cdot (\text{trace}(M))^2 \]

- ✅ Thresholds and detects corners across full image.
- 🎯 Produces corner-labeled image output.

---

## 🛠️ Building the Code
Use the root-level **Makefile** to compile all C++ source files across folders. Each subfolder has a main `.cpp` program you can run individually.

### Example:
```bash
make
./07_HarrisResponseFinal/harris_response ../01_Images/original_image.png
```

---

## 🧩 Summary
This modular project walks through:
1. Edge detection and gradients
2. Covariance matrix formation
3. Eigenvalue-based patch classification
4. Taylor series interpretation
5. Harris response computation

Each folder builds upon the previous, and the image outputs provide visual confirmation of each step.

---

Feel free to extend this project with:
- Real-time webcam corner detection
- Parameter tuning (block sizes, smoothing)
- Animation of gradient and surface approximations

Enjoy exploring the math and vision behind one of the most powerful classic algorithms in computer vision!

