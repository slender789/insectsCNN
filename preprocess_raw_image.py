import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def preprocess_image(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # 50 is a tune
  _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
  kernel = np.ones((3, 3), np.uint8)
  dilated = cv2.dilate(thresh, kernel, iterations=1)
  eroded = cv2.erode(dilated, kernel, iterations=1)
  equalized = cv2.equalizeHist(eroded)
  return equalized

image_path = 'IMG_5790_raw.JPG'
image = cv2.imread(image_path)
preprocessed = preprocess_image(image)

L = label(preprocessed)
props = regionprops(L)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

height, width = image.shape[:2]

for prop in props:
    centro = prop.centroid
    radio = prop.major_axis_length / 2
    if prop.area > 6000:
        circle = plt.Circle((centro[1], centro[0]), radio, color='b', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(centro[1], centro[0], '2', color='b')
    # areas are tune
    elif prop.area < 6000 and prop.area > 500:
        circle = plt.Circle((centro[1], centro[0]), radio, color='y', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(centro[1], centro[0], '1', color='y')