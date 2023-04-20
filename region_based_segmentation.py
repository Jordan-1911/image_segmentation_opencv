import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import data, morphology
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import watershed, mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb
import scipy.ndimage as nd

# Remove the following line as it is specific to Jupyter Notebook
# %matplotlib inline

# Load images and convert to grayscale
rocket = data.rocket()
rocket_wh = rgb2gray(rocket)

# Apply edge segmentation
# Plot Canny edge detection
edges = canny(rocket_wh)
plt.imshow(edges, interpolation='gaussian')
plt.title('Canny detector')
plt.show()

# Fill regions to perform edge segmentation
fill_im = nd.binary_fill_holes(edges)
plt.imshow(fill_im)
plt.title('Region Filling')
plt.show()

# Region Segmentation
# First, we print the elevation map
elevation_map = sobel(rocket_wh)
plt.imshow(elevation_map)
plt.show()

# Since, the contrast difference is not much. Anyways we will perform it
markers = np.zeros_like(rocket_wh)
markers[rocket_wh < 0.1171875] = 1  # 30/255
markers[rocket_wh > 0.5859375] = 2  # 150/255

plt.imshow(markers)
plt.title('markers')
plt.show()

# Perform watershed region segmentation
segmentation = watershed(elevation_map, markers)

plt.imshow(segmentation)
plt.title('Watershed segmentation')
plt.show()

# Plot overlays and contour
segmentation = nd.binary_fill_holes(segmentation - 1)
label_rock = label(segmentation)
# Overlay image with different labels
image_label_overlay = label2rgb(label_rock, image=rocket_wh)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16), sharey=True)
ax1.imshow(rocket_wh)
ax1.contour(segmentation, [0.8], linewidths=1.8, colors='w')
ax2.imshow(image_label_overlay)

plt.show()

