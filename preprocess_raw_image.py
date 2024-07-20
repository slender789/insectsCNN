import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    equalized = cv2.equalizeHist(eroded)
    return equalized

def store_image(image, output_dir, name, y, x, radio):
    minr = int(y - radio)
    minc = int(x - radio)
    maxr = int(y + radio)
    maxc = int(x + radio)

    # Ensure the bounding box is within the image boundaries
    minr = max(minr, 0)
    minc = max(minc, 0)
    maxr = min(maxr, image.shape[0])
    maxc = min(maxc, image.shape[1])

    # Crop the image
    cropped = image[minr:maxr, minc:maxc]

    # Save the cropped image
    output_path = os.path.join(output_dir, f'circle_{name}.png')
    cv2.imwrite(output_path, cropped)

def crop_image(image, y, x, radio):
    minr = int(y - radio)
    minc = int(x - radio)
    maxr = int(y + radio)
    maxc = int(x + radio)

    # Ensure the bounding box is within the image boundaries
    minr = max(minr, 0)
    minc = max(minc, 0)
    maxr = min(maxr, image.shape[0])
    maxc = min(maxc, image.shape[1])

    # Crop the image
    return image[minr:maxr, minc:maxc]

def countInsects(image_path, store=False, output_dir='counted_insects'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Failed to load image')

    if store:
        os.makedirs(output_dir, exist_ok=True)
    else:
        cropped_images = []

    preprocessed = preprocess_image(image)

    L = label(preprocessed)
    props = regionprops(L)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    for i, prop in enumerate(props):
        centro = prop.centroid
        radio = prop.major_axis_length / 2

        if 500 < prop.area <= 1500:
            circle = plt.Circle((centro[1], centro[0]), radio, color='m', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(centro[1], centro[0], '1.5-0.5', color='m')
            if store:
                store_image(image, output_dir, f'1.5-0.5_{i}', centro[0], centro[1], radio)
            else:
                cropped_images.append(crop_image(image, centro[0], centro[1], radio))
        if 1500 < prop.area <= 2500:
            circle = plt.Circle((centro[1], centro[0]), radio, color='c', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(centro[1], centro[0], '2.5-1.5', color='c')
            if store:
                store_image(image, output_dir, f'2.5-1.5_{i}', centro[0], centro[1], radio)
            else:
                cropped_images.append(crop_image(image, centro[0], centro[1], radio))
        if 2500 < prop.area <= 3500:
            circle = plt.Circle((centro[1], centro[0]), radio, color='b', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(centro[1], centro[0], '3.5-2.5', color='b')
            if store:
                store_image(image, output_dir, f'3.5-2.5_{i}', centro[0], centro[1], radio)
            else:
                cropped_images.append(crop_image(image, centro[0], centro[1], radio))
        if 3500 < prop.area <= 4500:
            circle = plt.Circle((centro[1], centro[0]), radio, color='y', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(centro[1], centro[0], '4.5-3.5', color='y')
            if store:
                store_image(image, output_dir, f'4.5-3.5_{i}', centro[0], centro[1], radio)
            else:
                cropped_images.append(crop_image(image, centro[0], centro[1], radio))
        if 4500 < prop.area <= 5500:
            circle = plt.Circle((centro[1], centro[0]), radio, color='g', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(centro[1], centro[0], '5.5-4.5', color='g')
            if store:
                store_image(image, output_dir, f'5.5-4.5_{i}', centro[0], centro[1], radio)
            else:
                cropped_images.append(crop_image(image, centro[0], centro[1], radio))
        if prop.area > 5500:
            circle = plt.Circle((centro[1], centro[0]), radio, color='r', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(centro[1], centro[0], '>=5.5', color='r')
            if store:
                store_image(image, output_dir, f'>=5.5_{i}', centro[0], centro[1], radio)
            else:
                cropped_images.append(crop_image(image, centro[0], centro[1], radio))

    if not store:
        return cropped_images

if __name__ == "__main__":
    cropped_images = countInsects('IMG_5784.JPG', store=False)
    for i, cropped_image in enumerate(cropped_images):
        cv2.imshow(f'Cropped Image {i}', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
