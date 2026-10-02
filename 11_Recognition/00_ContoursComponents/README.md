# Module 16 — Contours, Connected Components & Shape Descriptors

This lesson starts Part IV: **Classical Recognition and Tracking**.

Earlier modules answered questions such as:

- where are the edges?
- where are the corners?
- which local features correspond?
- how does geometry map points?
- how do features move over time?

Now we begin asking:

> Which foreground regions exist, what are their boundaries, and how can we describe their shape?

The canonical pipeline is:

~~~text
image
  |
  v
binary segmentation
  |
  +----------------------+
  |                      |
  v                      v
connected components     contours + hierarchy
  |                      |
  v                      v
region statistics        boundary geometry
  |                      |
  +----------+-----------+
             |
             v
       shape descriptors
             |
             v
     filtering / recognition
~~~

## Learning objectives

By the end of this lesson you should be able to:

- explain the difference between connected components and contours
- choose 4- vs 8-connectivity conceptually
- interpret connected-component label images
- compute area, bounding box, and centroid from region statistics
- extract contour hierarchy
- distinguish outer contours from holes
- compute contour area and perimeter
- compute centroid from image moments
- compute circularity
- compute aspect ratio and extent
- compute convex hull and solidity
- simplify a contour with Douglas-Peucker polygon approximation
- use polygon vertex count for simple geometric recognition
- understand Hu moment invariants
- understand which descriptors are scale/rotation/translation sensitive
- filter tiny regions before higher-level recognition
- export shape measurements for downstream pipelines

## 1. Binary segmentation comes first

Classical region analysis typically starts from a binary image:

~~~text
0   = background
255 = foreground
~~~

This lesson uses a fixed threshold for clarity.

For real images, the binary mask might come from:

- global thresholding
- Otsu thresholding
- adaptive thresholding
- color segmentation
- background subtraction
- semantic segmentation
- motion segmentation

The quality of the binary mask strongly affects every later descriptor.

## 2. Connected components

A connected-component algorithm assigns a unique integer label to each connected foreground region.

Conceptually:

~~~text
background = 0

object A = 1
object B = 2
object C = 3
...
~~~

The canonical implementation uses:

~~~text
connectedComponentsWithStats()
~~~

with 8-connectivity.

For every foreground component it collects:

- label ID
- pixel area
- axis-aligned bounding box
- centroid

## 3. Connectivity

Connectivity defines which neighboring pixels belong to the same region.

### 4-connectivity

Neighbors:

~~~text
up
down
left
right
~~~

Diagonal touching does not connect two regions.

### 8-connectivity

Adds the four diagonal neighbors.

The choice can change component count.

This lesson uses 8-connectivity because it is common for foreground-object labeling.

## 4. Region area

For connected components, area is simply the number of foreground pixels assigned to that label.

This is different from contour area, which is a geometric area derived from the contour boundary.

The two values are usually close for large regular objects but need not be identical because of discrete rasterization and boundary conventions.

## 5. Bounding box

For a component or contour, the axis-aligned bounding box is:

~~~text
x
y
width
height
~~~

It is useful for:

- cropping ROIs
- tracking
- object proposals
- filtering by size
- computing aspect ratio

## 6. Centroid

For connected components, OpenCV directly reports the centroid.

For contours, this lesson computes it from moments:

$$
c_x=\frac{m_{10}}{m_{00}}
$$

$$
c_y=\frac{m_{01}}{m_{00}}
$$

where:

$$
m_{00}
$$

corresponds to area.

## 7. Contours

A contour is an ordered boundary representation.

The lesson uses:

~~~text
findContours(..., RETR_CCOMP, CHAIN_APPROX_SIMPLE)
~~~

This gives both:

- boundary point sequences
- a two-level contour hierarchy

Contours are useful when boundary geometry matters.

Examples:

- shape recognition
- perimeter
- polygon approximation
- convexity
- hole detection

## 8. Connected components vs contours

They answer related but different questions.

### Connected components

Best for:

~~~text
which foreground pixels belong to the same region?
~~~

### Contours

Best for:

~~~text
what is the boundary geometry and topology of that region?
~~~

The synthetic lesson scene contains a donut.

Connected-component interpretation:

~~~text
one foreground object
~~~

Contour interpretation:

~~~text
one outer contour
one inner hole contour
~~~

That distinction is intentional.

## 9. Contour hierarchy

With CCOMP retrieval, a hole contour has a parent outer contour.

The hierarchy representation stores relationships such as:

~~~text
next sibling
previous sibling
first child
parent
~~~

The lesson identifies a contour as a hole when its parent index is non-negative.

This becomes useful for objects like:

- rings
- letters such as O/A/P
- washers
- nested regions
- document shapes

## 10. Perimeter

Contour perimeter is computed with:

~~~text
arcLength(contour, true)
~~~

For a closed contour:

$$
P = \sum_i \|p_{i+1}-p_i\|
$$

Perimeter is sensitive to:

- image resolution
- contour noise
- thresholding
- boundary pixelization

A noisy boundary can have a surprisingly large perimeter.

## 11. Circularity

The lesson computes:

$$
C =
\frac{4\pi A}{P^2}
$$

For an ideal circle:

$$
C=1
$$

Real rasterized circles are slightly below 1.

More elongated or irregular shapes have smaller circularity.

Circularity is scale-invariant because both numerator and denominator scale quadratically.

It is not a complete classifier by itself.

## 12. Aspect ratio

For an axis-aligned bounding box:

$$
AR=
\frac{width}{height}
$$

This is useful for distinguishing simple rectangles.

But it depends on object orientation.

A rotated rectangle can have a very different axis-aligned aspect ratio.

Later exercises can replace it with minAreaRect().

## 13. Extent

Extent measures how much of the bounding box is occupied by the contour:

$$
extent=
\frac{contour\ area}{bounding\ box\ area}
$$

A filled axis-aligned rectangle has high extent.

A circle has lower extent because the corners of its bounding box are empty.

## 14. Convex hull and solidity

The convex hull is the smallest convex polygon containing the contour.

Solidity is:

$$
solidity=
\frac{contour\ area}{convex\ hull\ area}
$$

A convex shape has solidity near 1.

Strong concavities reduce solidity.

This is useful for:

- hand/gesture shapes
- defect analysis
- irregular parts
- distinguishing convex vs concave objects

## 15. Polygon approximation

Contours often contain many rasterized boundary points.

Douglas-Peucker approximation simplifies them.

The lesson uses:

$$
\epsilon =
fraction \times perimeter
$$

Default:

~~~text
epsilon fraction = 0.02
~~~

A simple triangle should reduce to roughly 3 vertices.

A rectangle should reduce to roughly 4.

A circle requires many more.

## 16. Simple shape classification

This lesson intentionally uses a small transparent heuristic:

~~~text
3 vertices -> triangle
4 vertices -> rectangle / square-like
high circularity + high solidity -> circle-like
otherwise -> other
~~~

This is not presented as a universal classifier.

The goal is to show how handcrafted descriptors become a feature pipeline.

## 17. Hu moments

Hu moments are seven combinations of normalized central moments designed to be invariant to:

- translation
- scale
- rotation

under the ideal continuous formulation.

The raw values can span many orders of magnitude.

The lesson stores a signed log transform for readability.

Hu moments are useful for classical shape comparison, but they can still be sensitive to:

- segmentation noise
- rasterization
- deformation
- partial occlusion

## 18. Synthetic scene

Without an input image, the lesson generates:

- one filled circle
- one filled rectangle
- one filled triangle
- one donut
- two tiny noise-like components

This provides deterministic coverage of:

- multiple connected regions
- contour hierarchy
- a hole
- area filtering
- polygon approximation
- descriptor classification

The default contour filter:

~~~text
--min-area 100
~~~

removes the two tiny foreground components from shape recognition while leaving them visible in the connected-component analysis.

## 19. Outputs

The executable writes:

~~~text
01_binary.png
02_connected_components.png
03_contours_and_shapes.png
shape_descriptors.csv
~~~

### 01_binary.png

The segmentation mask used by both analysis paths.

### 02_connected_components.png

Every foreground label receives a deterministic color.

The visualization includes:

- component bounding boxes
- centroids
- label IDs
- pixel areas

### 03_contours_and_shapes.png

Displays:

- green outer contours
- red hole contours
- bounding boxes
- centroids
- simple shape labels

### shape_descriptors.csv

Exports:

- contour index
- hole flag
- predicted simple shape
- area
- perimeter
- centroid
- bounding box
- circularity
- aspect ratio
- extent
- solidity
- polygon vertex count
- seven log-transformed Hu moments

## 20. Deterministic self-test

Run:

~~~bash
./build/cv9x_shape_analysis --self-test
ctest --test-dir build --output-on-failure
~~~

The self-test verifies:

- six foreground components exist before area filtering
- four large semantic objects remain above 100 pixels
- four large outer contours are analyzed
- exactly one hole contour is retained
- the circle is classified as circle-like
- the circle has high circularity
- the rectangle approximates to four vertices
- the triangle approximates to three vertices
- the donut outer boundary remains circle-like
- circle centroid is recovered near ground truth
- Hu moments remain invariant when the same rectangle is translated

## 21. Real-image mode

Run:

~~~bash
./build/cv9x_shape_analysis \
  --input parts.png \
  --output-dir build/shapes \
  --threshold 127 \
  --min-area 250 \
  --epsilon 0.02
~~~

If foreground is dark and background is bright:

~~~text
--invert
~~~

The lesson intentionally keeps segmentation simple so shape analysis remains the focus.

## Common failure cases

### Poor thresholding

If objects merge in the binary mask, connected components cannot magically separate them.

### Tiny noise regions

Filter by area or clean the mask morphologically.

### Touching objects

Connected components may merge them into one region.

Watershed or other separation methods may be needed.

### Treating contour area as pixel count

Contour area and component pixel area use different definitions.

### Using only one descriptor

Circularity, aspect ratio, or Hu moments alone are rarely sufficient for robust recognition.

### Ignoring holes

Topology can be discriminative.

### Over-aggressive polygon approximation

A large epsilon can turn curved shapes into incorrect low-vertex polygons.

### Assuming axis-aligned aspect ratio is rotation invariant

It is not.

## Engineering notes

A production region/shape record might include:

~~~text
region_id
frame_id
timestamp
pixel_area
bbox
centroid
contour
hole_count
perimeter
circularity
solidity
orientation
Hu moments
classification
confidence
segmentation source/version
~~~

For high-rate pipelines, consider whether full contours are required downstream.

A contour can contain hundreds or thousands of points, while many consumers may only need:

- bounding box
- centroid
- area
- selected descriptors

Avoid carrying unnecessary geometry across process/module boundaries.

## Build

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cv9x_shape_analysis
~~~

## Synthetic demo

~~~bash
./build/cv9x_shape_analysis \
  --output-dir build/shape_analysis
~~~

## Exercises

1. Switch between 4- and 8-connectivity and create diagonally touching regions.
2. Add morphological opening/closing before component labeling.
3. Add minAreaRect() and rotation-invariant rectangle measurements.
4. Add convexity defects for concave shapes.
5. Use matchShapes() and compare it with direct Hu-moment distance.
6. Add a second donut and count holes per outer contour.
7. Create touching circles and separate them with watershed.
8. Track a connected component across video using the optical-flow concepts from Module 15.
9. Build a classical part-inspection classifier from area, aspect ratio, circularity, and solidity.

## Next lesson

The next module is **HOG + SVM object detection**.

That moves from direct region geometry into sliding-window appearance descriptors and a trained linear classifier.
